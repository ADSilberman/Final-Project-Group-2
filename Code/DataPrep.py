import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

'''
Shared data preparation for the NLP final project (Group 2).
Run once:  python data_prep.py
Creates:   data/train.csv  and  data/test.csv
Everyone loads these two files, so all models use the same split.
'''

# ----------------------------- settings (agree on these as a team) -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_SAMPLES_PER_CLASS = 50          # drop specialties with fewer notes than this

# These labels describe a document type, not a medical specialty
DROP_CLASSES = ['SOAP / Chart / Progress Notes', 'Discharge Summary', 'Consult - History and Phy.',
                'Office Notes', 'Letters', 'IME-QME-Work Comp etc.', 'Emergency Room Reports']

KEEP_COLUMNS = ['sample_name', 'transcription', 'description', 'medical_specialty', 'keywords',
                'transcription_length', 'complexity_score']

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

# each transcription appears once per instruction type -> keep one row per note
df = df.drop_duplicates(subset='transcription').reset_index(drop=True)
print(f'unique transcriptions: {df.shape[0]}')

# ----------------------------- 3. filter labels -----------------------------
df = df[~df['medical_specialty'].isin(DROP_CLASSES)]

counts = df['medical_specialty'].value_counts()
keep = counts[counts >= MIN_SAMPLES_PER_CLASS].index
df = df[df['medical_specialty'].isin(keep)].reset_index(drop=True)

print(f'rows after filtering: {df.shape[0]}')
print(f'number of classes: {df["medical_specialty"].nunique()}')
print(df['medical_specialty'].value_counts())

# ----------------------------- 4. extra label for the length task -----------------------------
# 1 = longer than average, 0 = shorter (average taken from the whole cleaned set)
df['is_long'] = (df['transcription_length'] > df['transcription_length'].mean()).astype(int)

# ----------------------------- 5. split -----------------------------
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                     stratify=df['medical_specialty'])
print(f'shape of train data is {train_df.shape}')
print(f'shape of test data is {test_df.shape}')

# ----------------------------- 6. save -----------------------------
os.makedirs(OUT_DIR, exist_ok=True)
train_df.to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)
test_df.to_csv(os.path.join(OUT_DIR, 'test.csv'), index=False)
print(f'saved to {OUT_DIR}/train.csv and {OUT_DIR}/test.csv')