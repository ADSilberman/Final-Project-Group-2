import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

'''
Shared data preparation for the NLP final project (Group 2).
Run once:  python DataPrep.py
Creates:   data/train.csv  and  data/test.csv

Each file has ALL task rows (instruction + task_output) for its notes.
The split is done by note (transcription), so all tasks of a note are on the same side.
Everyone filters the rows they need, e.g.:
    train = pd.read_csv('data/train.csv')
    summ = train[train['instruction'].str.contains('summar', case=False)]
For one row per note (classification):
    notes = train.drop_duplicates(subset='transcription')
'''

# ----------------------------- settings (agree on these as a team) -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_SAMPLES_PER_CLASS = 50          # drop specialties with fewer notes than this

# These labels describe a document type, not a medical specialty
DROP_CLASSES = ['SOAP / Chart / Progress Notes', 'Discharge Summary', 'Consult - History and Phy.',
                'Office Notes', 'Letters', 'IME-QME-Work Comp etc.', 'Emergency Room Reports']

KEEP_COLUMNS = ['sample_name', 'instruction', 'task_output', 'transcription', 'description',
                'medical_specialty', 'keywords', 'derived_keywords', 'transcription_length', 'normalized_length',
                'complexity_score']

OUT_DIR = 'data'

# ----------------------------- 1. load -----------------------------
df = load_dataset('DataFog/medical-transcription-instruct')['train'].to_pandas()
print(f'raw rows: {df.shape[0]}')

# ----------------------------- 2. clean -----------------------------
df = df[KEEP_COLUMNS]
df = df.dropna(subset=['transcription', 'medical_specialty'])

df['transcription'] = df['transcription'].str.strip()
df['medical_specialty'] = df['medical_specialty'].str.strip()
df = df[df['transcription'] != '']
df = df.drop_duplicates().reset_index(drop=True)      # remove exact duplicate rows only

# ----------------------------- 3. filter labels -----------------------------
df = df[~df['medical_specialty'].isin(DROP_CLASSES)]

# count classes by unique notes (not by rows, since each note has several tasks)
notes = df.drop_duplicates(subset='transcription')
counts = notes['medical_specialty'].value_counts()
keep = counts[counts >= MIN_SAMPLES_PER_CLASS].index
df = df[df['medical_specialty'].isin(keep)].reset_index(drop=True)

print(f'rows after filtering: {df.shape[0]}')
print(f'unique notes: {df["transcription"].nunique()}')
print(f'number of classes: {df["medical_specialty"].nunique()}')
print(df.drop_duplicates(subset="transcription")['medical_specialty'].value_counts())
print(df['instruction'].value_counts().head(10))

# ----------------------------- 4. extra label for the length task -----------------------------
# 1 = longer than average, 0 = shorter (average over unique notes)
avg_len = df.drop_duplicates(subset='transcription')['transcription_length'].mean()
df['is_long'] = (df['transcription_length'] > avg_len).astype(int)

# ----------------------------- 5. split by note -----------------------------
notes = df.drop_duplicates(subset='transcription')
train_notes, test_notes = train_test_split(notes['transcription'], test_size=TEST_SIZE,
                                           random_state=RANDOM_STATE, stratify=notes['medical_specialty'])

train_df = df[df['transcription'].isin(set(train_notes))].reset_index(drop=True)
test_df = df[df['transcription'].isin(set(test_notes))].reset_index(drop=True)

print(f'shape of train data is {train_df.shape}  ({train_df["transcription"].nunique()} notes)')
print(f'shape of test data is {test_df.shape}  ({test_df["transcription"].nunique()} notes)')

# ----------------------------- 6. save -----------------------------
os.makedirs(OUT_DIR, exist_ok=True)
train_df.to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)
test_df.to_csv(os.path.join(OUT_DIR, 'test.csv'), index=False)
print(f'saved to {OUT_DIR}/train.csv and {OUT_DIR}/test.csv')