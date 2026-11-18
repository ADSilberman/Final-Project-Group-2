import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

'''
Shared Data preparation for the NLP final project (Group 2).
Run once:  python DataPrep.py
Creates:   Data/train.csv  and  Data/test.csv

Each file has ALL notes and ALL task rows (instruction + task_output), one row per note per task.
The split is done by note (transcription), so all tasks of a note are on the same side.

medical_specialty = the ONE specialty label used for classification.
It is EMPTY for notes that don't pass the classification filters
(document types like "Discharge Summary", or specialties with < 50 notes).

How to use:
    train = pd.read_csv('Data/train.csv')

    # generation (all notes), e.g. summaries
    summ = train[train['instruction'].str.contains('summar', case=False)]

    # classification (labelled notes only, one row per note)
    clf = train.dropna(subset=['medical_specialty']).drop_duplicates(subset='transcription')
'''

# ----------------------------- settings (agree on these as a team) -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_SAMPLES_PER_CLASS = 50          # specialties with fewer notes get no classification label

# These labels describe a document type, not a medical specialty
DROP_CLASSES = ['SOAP / Chart / Progress Notes', 'Discharge Summary', 'Consult - History and Phy.',
                'Office Notes', 'Letters', 'IME-QME-Work Comp etc.', 'Emergency Room Reports']

# Broad catch-all labels: only used when a note has no more specific label
GENERIC = ['Surgery', 'General Medicine']

KEEP_COLUMNS = ['sample_name', 'instruction', 'task_output', 'transcription', 'description',
                'medical_specialty', 'keywords', 'derived_keywords', 'transcription_length', 'normalized_length',
                'complexity_score']

OUT_DIR = 'Data'

# ----------------------------- 1. load -----------------------------
df = load_dataset('DataFog/medical-transcription-instruct')['train'].to_pandas()
print(f'raw rows: {df.shape[0]}')

# ----------------------------- 2. clean -----------------------------
df = df[KEEP_COLUMNS]
df = df.dropna(subset=['transcription'])

df['transcription'] = df['transcription'].str.strip()
df['medical_specialty'] = df['medical_specialty'].str.strip()
df = df[df['transcription'] != '']
df = df.drop_duplicates().reset_index(drop=True)      # remove exact duplicate rows only
print(f'unique notes: {df["transcription"].nunique()}')

# ----------------------------- 3. one classification label per note -----------------------------
# Some notes are listed under several specialties (e.g. Orthopedic and Surgery). Reasoning:
#  - the task is single-label classification, so each note needs one true label
#  - the same text with two labels confuses training and makes F1/Kappa unfair
# Rule for choosing:
#  1. ignore document-type labels (DROP_CLASSES): they are not medical specialties
#  2. prefer specific specialties over Surgery / General Medicine (broad catch-all labels)
#  3. among specific labels, prefer the smaller class (reduces class imbalance)
#  4. any remaining tie keeps the dataset order (kind='stable')
# These filters only affect the CLASSIFICATION label. Every note stays in the Data for generation.
labels = df[~df['medical_specialty'].isin(DROP_CLASSES)][['transcription', 'medical_specialty']].drop_duplicates()
note_counts = labels.groupby('medical_specialty')['transcription'].nunique()   # notes per specialty
labels['_generic'] = labels['medical_specialty'].isin(GENERIC)
labels['_size'] = labels['medical_specialty'].map(note_counts)

chosen = (labels.sort_values(['_generic', '_size'], kind='stable')
                .drop_duplicates(subset='transcription')[['transcription', 'medical_specialty']])

# specialties with too few notes get no label (too few examples to learn from or test fairly)
counts = chosen['medical_specialty'].value_counts()
keep = counts[counts >= MIN_SAMPLES_PER_CLASS].index
chosen = chosen[chosen['medical_specialty'].isin(keep)]

# ----------------------------- 4. one row per note per task -----------------------------
# a note listed under 2 specialties had every task twice -> keep one copy of each task,
# preferring the copy from the chosen specialty
chosen_map = dict(zip(chosen['transcription'], chosen['medical_specialty']))
df['_label'] = df['transcription'].map(chosen_map)                  # NaN if the note has no label
df['_is_chosen'] = df['medical_specialty'] == df['_label']
df = (df.sort_values('_is_chosen', ascending=False, kind='stable')
        .drop_duplicates(subset=['transcription', 'instruction']))

df['medical_specialty'] = df['_label']                              # the final classification label
df = df.drop(columns=['_label', '_is_chosen']).sort_index().reset_index(drop=True)

notes = df.drop_duplicates(subset='transcription')
print(f'rows: {df.shape[0]}  |  notes: {notes.shape[0]}  |  '
      f'notes with a classification label: {notes["medical_specialty"].notna().sum()}')
print(f'number of classes: {notes["medical_specialty"].nunique()}')
print(notes['medical_specialty'].value_counts())
print(df['instruction'].value_counts().head(10))

# ----------------------------- 5. extra label for the length task -----------------------------
# 1 = longer than average, 0 = shorter (average over unique notes)
avg_len = notes['transcription_length'].mean()
df['is_long'] = (df['transcription_length'] > avg_len).astype(int)

# ----------------------------- 6. split by note -----------------------------
# stratify by label so every specialty is in train and test; unlabelled notes form their own group
strata = notes['medical_specialty'].fillna('NO_LABEL')
train_notes, test_notes = train_test_split(notes['transcription'], test_size=TEST_SIZE,
                                           random_state=RANDOM_STATE, stratify=strata)

train_df = df[df['transcription'].isin(set(train_notes))].reset_index(drop=True)
test_df = df[df['transcription'].isin(set(test_notes))].reset_index(drop=True)

print(f'shape of train Data is {train_df.shape}  ({train_df["transcription"].nunique()} notes)')
print(f'shape of test Data is {test_df.shape}  ({test_df["transcription"].nunique()} notes)')

# ----------------------------- 7. save -----------------------------
os.makedirs(OUT_DIR, exist_ok=True)
train_df.to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)
test_df.to_csv(os.path.join(OUT_DIR, 'test.csv'), index=False)
print(f'saved to {OUT_DIR}/train.csv and {OUT_DIR}/test.csv')