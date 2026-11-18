#%%
import pandas as pd
import os
import sys
from datasets import load_dataset

#%%
# CODE_DIR = os.path.join(os.getcwd(), 'Code')
# sys.path.append(CODE_DIR)
# DATA_DIR = os.path.join(CODE_DIR, 'Data')
# df = pd.read_csv(f'{DATA_DIR}/test.csv')
# print(df.head())
df = load_dataset('DataFog/medical-transcription-instruct')['train'].to_pandas()
print(df.head())

#%%
for col in df.columns:
    print(col)

#%%
"""There are 2358 total unique transcriptions"""
print(len(df['transcription'].unique()))

#%%
print(df['instruction'].unique())

#%%
"""Drop classes that don't have enough representation in the data and that are not a medical specialty"""
classification_data = df[df['instruction'] == 'Identify the medical specialty for this transcription']
print(classification_data['medical_specialty'].value_counts())

#%%
classification_data['medical_specialty'] = classification_data['medical_specialty'].str.strip()
classes_to_drop = ['SOAP / Chart / Progress Notes', 'Discharge Summary', 'Consult - History and Phy.',
                'Office Notes', 'Letters', 'IME-QME-Work Comp etc.', 'Emergency Room Reports']
classification_data = classification_data[~classification_data['medical_specialty'].isin(classes_to_drop)]
print(classification_data['medical_specialty'].value_counts())

#%%
min_samples = 50
counts = classification_data['medical_specialty'].value_counts()
keep = counts[counts >= min_samples].index
classification_data = classification_data[classification_data['medical_specialty'].isin(keep)].reset_index(drop=True)
print(len(classification_data['medical_specialty'].value_counts()))

#%%

