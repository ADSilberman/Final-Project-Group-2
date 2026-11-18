# ------------------------Libraries------------------------------------
import os
import sys
import json
import numpy as np
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from joblib import dump

# shared evaluation file lives one folder up (Code/Evaluate.py)
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CODE_DIR)
from Evaluate import evaluate_classification

DATA_DIR = os.path.join(CODE_DIR, 'data')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))     # saves models next to this script
# ----------------------------------------------------------------------
# Task: classify a medical transcription into its medical specialty.
# Three models, all evaluated with the shared Evaluate.py:
#   1. Rule-based : keyword lists per specialty + spaCy PhraseMatcher, predict the specialty
#                   with the highest (normalized) match score. No training.
#   2. Classical  : TF-IDF word 1-2 grams + TF-IDF char 3-5 grams (FeatureUnion) + LinearSVC.
#   3. Hybrid     : same as Classical, plus the rule-based match scores as extra features,
#                   so the SVM learns how much to trust each rule.
# Keyword lists = manual medical keywords + words learned from the TRAIN notes only.
# ----------------------------------------------------------------------

# Manual keywords: edit/extend these. Keys must match the specialty labels exactly
# (any key that isn't a label in the data is ignored).
MANUAL_KEYWORDS = {
    'Cardiovascular / Pulmonary': ['chest pain', 'atrial fibrillation', 'coronary', 'ecg', 'ekg', 'stent',
                                   'catheterization', 'ejection fraction', 'shortness of breath', 'copd'],
    'Orthopedic': ['fracture', 'knee', 'shoulder', 'arthroplasty', 'tendon', 'ligament', 'hip', 'spine'],
    'Neurology': ['seizure', 'headache', 'stroke', 'mri of the brain', 'neuropathy', 'tremor', 'eeg'],
    'Gastroenterology': ['colonoscopy', 'endoscopy', 'abdominal pain', 'colon', 'polyp', 'gerd', 'esophagus'],
    'Urology': ['bladder', 'prostate', 'kidney stone', 'cystoscopy', 'urinary', 'hematuria'],
    'Obstetrics / Gynecology': ['pregnancy', 'uterus', 'cesarean', 'gestation', 'ovary', 'vaginal'],
    'Radiology': ['ct scan', 'x-ray', 'ultrasound', 'impression', 'contrast'],
    'Nephrology': ['renal failure', 'dialysis', 'creatinine', 'kidney'],
    'Ophthalmology': ['cataract', 'retina', 'visual acuity', 'intraocular', 'cornea'],
    'ENT - Otolaryngology': ['tonsillectomy', 'sinus', 'ear', 'nasal', 'larynx'],
}

N_AUTO_KEYWORDS = 15        # words learned per specialty from train notes
RULE_WEIGHT = 1.0           # how strongly rule features count in the hybrid model


# ------------------------Data------------------------------------------
def read_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    # one row per note for classification (files have one row per task)
    train_df = train_df.drop_duplicates(subset='transcription')[['transcription', 'medical_specialty']]
    test_df = test_df.drop_duplicates(subset='transcription')[['transcription', 'medical_specialty']]
    train_df['transcription'] = train_df['transcription'].astype(str)
    test_df['transcription'] = test_df['transcription'].astype(str)

    print(f'shape of train data is {train_df.shape}')
    print(f'shape of test data is {test_df.shape}')
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ------------------------Rule-based part-------------------------------
def learn_keywords(texts, labels, n=N_AUTO_KEYWORDS):
    '''Find the most distinctive words for each specialty in the TRAIN notes.
    Score = (share of notes in this class with the word) - (share of notes in other classes with it).'''
    cv = CountVectorizer(stop_words='english', binary=True, min_df=5, token_pattern=r'(?u)\b[a-zA-Z]{3,}\b')
    X = cv.fit_transform(texts)
    vocab = np.array(cv.get_feature_names_out())
    labels = np.array(labels)

    keywords = {}
    for c in np.unique(labels):
        in_c = labels == c
        p_in = np.asarray(X[in_c].mean(axis=0)).ravel()
        p_out = np.asarray(X[~in_c].mean(axis=0)).ravel()
        score = p_in - p_out
        top = [i for i in np.argsort(score)[::-1][:n] if score[i] > 0]   # only words more common in this class
        keywords[str(c)] = [str(vocab[i]) for i in top]
    return keywords


def build_keywords(texts, labels):
    keywords = learn_keywords(texts, labels)
    for c, words in MANUAL_KEYWORDS.items():
        if c in keywords:
            keywords[c] = list(dict.fromkeys(words + keywords[c]))   # manual first, no duplicates
    return keywords


class RuleFeatures(BaseEstimator, TransformerMixin):
    '''Turns each note into one score per specialty = share of that specialty's keywords found in the note.
    fit() builds the keyword lists from the training notes only.'''

    def fit(self, X, y=None):
        self.keywords_ = build_keywords(list(X), list(y))
        self.classes_ = sorted(self.keywords_)
        return self

    def _matcher(self, nlp):
        matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
        for c in self.classes_:
            matcher.add(c, [nlp.make_doc(w) for w in self.keywords_[c]])
        return matcher

    def transform(self, X):
        nlp = spacy.blank('en')                  # tokenizer only, no model download needed
        nlp.max_length = 2_000_000
        matcher = self._matcher(nlp)
        col = {c: i for i, c in enumerate(self.classes_)}

        scores = np.zeros((len(X), len(self.classes_)))
        for row, doc in enumerate(nlp.pipe(list(X), batch_size=50)):
            found = {}
            for match_id, start, end in matcher(doc):
                c = nlp.vocab.strings[match_id]
                found.setdefault(c, set()).add(doc[start:end].text.lower())
            for c, words in found.items():
                scores[row, col[c]] = len(words) / len(self.keywords_[c])
        return csr_matrix(scores * RULE_WEIGHT)


def rule_based_predict(rules, texts, fallback):
    scores = rules.transform(texts).toarray()
    preds = np.array(rules.classes_)[scores.argmax(axis=1)]
    preds[scores.max(axis=1) == 0] = fallback      # nothing matched -> most common class
    return preds


# ------------------------Classical / Hybrid part-----------------------
def make_vectorizer(use_rules=False):
    parts = [
        ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents='unicode')),
        ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                 max_features=300000)),
    ]
    if use_rules:
        parts.append(('rules', RuleFeatures()))
    return FeatureUnion(parts)


def model_training(train_df, use_rules=False):
    vectorizer = make_vectorizer(use_rules)
    X_train = vectorizer.fit_transform(train_df['transcription'], train_df['medical_specialty'])
    model = LinearSVC(C=0.3, random_state=42)
    model.fit(X_train, train_df['medical_specialty'])
    return model, vectorizer


def model_testing(model, vectorizer, test_df):
    X_test = vectorizer.transform(test_df['transcription'])
    return model.predict(X_test)


# ------------------------Saving----------------------------------------
def save(obj, name):
    dump(obj, os.path.join(OUT_DIR, name))


def save_keywords(keywords):
    with open(os.path.join(OUT_DIR, 'keywords.json'), 'w') as f:
        json.dump(keywords, f, indent=2)


# ----------------------------------------------------------------------
def main():
    train_df, test_df = read_data()
    y_test = test_df['medical_specialty']

    # 1. Rule-based
    rules = RuleFeatures().fit(train_df['transcription'], train_df['medical_specialty'])
    for c in rules.classes_[:5]:
        print(f'{c}: {rules.keywords_[c][:10]}')
    majority = train_df['medical_specialty'].mode()[0]
    rule_preds = rule_based_predict(rules, test_df['transcription'], fallback=majority)
    evaluate_classification(y_test, rule_preds, model_name='Rule-Based', task='specialty')
    save_keywords(rules.keywords_)

    # 2. Classical (TF-IDF + LinearSVC)
    model, vectorizer = model_training(train_df, use_rules=False)
    preds = model_testing(model, vectorizer, test_df)
    evaluate_classification(y_test, preds, model_name='TF-IDF + LinearSVC', task='specialty')
    save(model, 'classical_model.joblib')
    save(vectorizer, 'classical_vectorizer.joblib')

    # 3. Hybrid (TF-IDF + rule features + LinearSVC)
    model, vectorizer = model_training(train_df, use_rules=True)
    preds = model_testing(model, vectorizer, test_df)
    evaluate_classification(y_test, preds, model_name='Hybrid (Rules + TF-IDF + LinearSVC)', task='specialty')
    save(model, 'hybrid_model.joblib')
    save(vectorizer, 'hybrid_vectorizer.joblib')


if __name__ == '__main__':
    main()