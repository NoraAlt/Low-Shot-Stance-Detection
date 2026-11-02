import pandas as pd
import re
import json
import os

def clean_arabic_text(text):
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    text = re.sub(r'URL', '', text)
    text = re.sub(r'http', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'@USER', '', text)
    text = re.sub(r'@user', '', text)
    text = re.sub(r'MENTION','',text)
    
    # Remove extra white spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters
    text = re.sub(r'<LF>', '', text)
    text = re.sub(r'<LF', '', text)
    text = re.sub(r'RT', '', text)
    text = re.sub(r'[÷×؛<>*&^%{}~¦+|]', '', text)
    
    # Remove repeating characters that are more than 3
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    
    # Normalize Arabic letters
    text = re.sub("[إآ]", "ا", text)
    text = re.sub("گ", "ك", text)
    
    # Remove al-harakat
    noise = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil
                         """, re.VERBOSE)
    text = re.sub(noise, '', text)
    return text


# Set the directory path to the folder that contains your JSON files
dir_path = '/home/nora_alturayeif/STANCE/My_T/data/JSON'

# Change the current working directory to the directory path
os.chdir(dir_path)

# Define the file paths of my JSON files (31 files)
files_stance = ['AraCovidVac_test.json','AraCovidVac_traindev.json', 'Mawqif_stance_test.json','Mawqif_stance_train.json']
files_NLI = ['XGLUE-xnli_train.json','XGLUE-xnli_test.json', 'AraStance_test.json','AraStance_traindev.json', 'Headlines_test.json','Headlines_traindev.json']
files_OFF = ['OSACT2022_test.json', 'OSACT2022_traindev.json', 'OSACT2020_test.json', 'OSACT2020_traindev.json']
files_sarcasm = ['ArSarcasm_sarcastic_test.json','ArSarcasm_sarcastic_train.json', 'Mawqif_sarcasm_test.json','Mawqif_sarcasm_train.json',
                 'iSarcasmEval_test.json','iSarcasmEval_train.json']
files_sentiment = ['AJGT_train.json', 'ASTD_test.json','ASTD_traindev.json', 'MASC_train.json','MASC_test.json', 'UMSAB_test.json', 'UMSAB_traindev.json', 
                   'Mawqif_sentiment_test.json','Mawqif_sentiment_train.json', 'ArSarcasm_sentiment_train.json','ArSarcasm_sentiment_test.json']


file_paths = ['AJGT_train.json','AraCovidVac_test.json', 'AraCovidVac_traindev.json', 'AraStance_test.json', 'AraStance_traindev.json',
              'ArSarcasm_sarcastic_test.json', 'ArSarcasm_sarcastic_train.json', 'ArSarcasm_sentiment_test.json', 'ArSarcasm_sentiment_train.json',
              'ASTD_test.json', 'ASTD_traindev.json', 'Headlines_test.json', 'Headlines_traindev.json', 'iSarcasmEval_test.json', 'iSarcasmEval_train.json',
              'MASC_train.json', 'MASC_test.json', 'Mawqif_sarcasm_test.json', 'Mawqif_sarcasm_train.json', 'Mawqif_sentiment_test.json', 'Mawqif_sentiment_train.json',
              'Mawqif_stance_test.json', 'Mawqif_stance_train.json', 'OSACT2022_test.json', 'OSACT2022_traindev.json', 'OSACT2020_test.json', 'OSACT2020_traindev.json',
              'UMSAB_test.json', 'UMSAB_traindev.json', 'XGLUE-xnli_train.json', 'XGLUE-xnli_test.json']


train_dfs = []
val_dfs = []
# Loop through each file path
for i, file_path in enumerate(file_paths):
    # Read the JSON file into a pandas dataframe
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)

    
    if 'context' in df.columns:
        print(f"Processing file {i+1}: {file_path} with size: {df.shape}")
        # Clean the text columns
        df['context'] = df['context'].apply(clean_arabic_text)
        # Keep onlt the needed columns
        df = df[['context', 'response']]
    if 's1' in df.columns:
        print(f"Processing file {i+1}: {file_path} with size: {df.shape}")
        # Clean the text columns
        df['s1'] = df['s1'].apply(clean_arabic_text)
        df['s2'] = df['s2'].apply(clean_arabic_text)
        # Concatnate
        df['context'] = df['s1'] + "separator" + df['s2']
        # Keep onlt the needed columns
        df = df[['context', 'response']]

    # Add the cleaned dataframe to the list of dataframes
    if "train" in file_path or "traindev" in file_path:
        train_dfs.append(df)
    elif "test" in file_path:
        val_dfs.append(df)

# Concatenate the dataframes into a single dataframe before calling to_json on it.
train_d = pd.concat(train_dfs, ignore_index=True)
val_d = pd.concat(val_dfs, ignore_index=True)

# Save the cleaned dataframe to a new JSON file
new_dir_path = '/home/nora_alturayeif/STANCE/My_T/data/data_for_train'
os.chdir(new_dir_path)

train_d.to_json("train.json", orient='records', lines=True, force_ascii=False)
print("Size of training data after merging:",train_d.shape)

val_d.to_json("test.json", orient='records', lines=True, force_ascii=False)
print("Size of validation data after merging:",val_d.shape)
