import os
import pandas as pd
from sklearn import metrics

'''
Shared evaluation for the NLP final project (Group 2).
Every model calls these functions so the scores are computed the same way.

Usage (from your model's script):
    import sys; sys.path.append('..')          # if your script is inside a subfolder
    from evaluate import evaluate_classification, evaluate_generation

    evaluate_classification(test_df['medical_specialty'], y_pred, model_name='Rule-Based', task='specialty')
    evaluate_generation(test_df['description'], summaries, model_name='Rule-Based', task='summary')

Each call prints the scores and adds one row to Results/Results.csv.
'''

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results', 'Results.csv')


def save_result(row):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    new = pd.DataFrame([row])
    if os.path.exists(RESULTS_FILE):
        new = pd.concat([pd.read_csv(RESULTS_FILE), new], ignore_index=True)
    new.to_csv(RESULTS_FILE, index=False)


# ----------------------------- classification -----------------------------
def evaluate_classification(y_true, y_pred, model_name, task='specialty', show_report=True):
    y_true, y_pred = list(y_true), list(y_pred)

    scores = {
        'model': model_name,
        'task': task,
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'f1_macro': metrics.f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'cohen_kappa': metrics.cohen_kappa_score(y_true, y_pred),
    }

    print(f'\n===== {model_name} | {task} =====')
    if show_report:
        print(metrics.classification_report(y_true, y_pred, zero_division=0))
        print(metrics.confusion_matrix(y_true, y_pred))
    for k in ['accuracy', 'f1_macro', 'f1_weighted', 'cohen_kappa']:
        print(f'{k}: {scores[k]:.4f}')

    save_result(scores)
    return scores


# ----------------------------- generation -----------------------------
def evaluate_generation(references, predictions, model_name, task='summary'):
    # pip install rouge-score nltk
    import nltk
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score

    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    references = [str(r) for r in references]
    predictions = [str(p) for p in predictions]
    ref_tokens = [r.lower().split() for r in references]
    pred_tokens = [p.lower().split() for p in predictions]

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    for r, p in zip(references, predictions):
        s = scorer.score(r, p)
        for k in rouge:
            rouge[k] += s[k].fmeasure
    rouge = {k: v / len(references) for k, v in rouge.items()}

    # BLEU (smoothing so short texts don't get 0)
    bleu = corpus_bleu([[r] for r in ref_tokens], pred_tokens,
                       smoothing_function=SmoothingFunction().method1)

    # METEOR
    meteor = sum(meteor_score([r], p) for r, p in zip(ref_tokens, pred_tokens)) / len(references)

    scores = {'model': model_name, 'task': task, 'bleu': bleu, 'meteor': meteor, **rouge}

    print(f'\n===== {model_name} | {task} =====')
    for k in ['bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL']:
        print(f'{k}: {scores[k]:.4f}')

    save_result(scores)
    return scores


# ----------------------------- show all Results -----------------------------
if __name__ == '__main__':
    if os.path.exists(RESULTS_FILE):
        print(pd.read_csv(RESULTS_FILE).to_string(index=False))
    else:
        print('No Results yet.')