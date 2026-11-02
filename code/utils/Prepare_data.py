## This file will implement 3 steps: 1- clean text, 2- prompting, 3- merging

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

#------------------------------------------------------------------

def prompting(data,task):

    language = "En" #"En" "Ar_En" "Ar"

    if language == 'Ar_En':
        prompt_stance = 'How would you describe the stance used in this text regarding: {target}? Text: {context}. Possible answers: neutral (محايد), against (معارض), in favor (مؤيد)'
        prompt_nli= 'Does sentence 1 entail (يوافق), contradict (يناقض), or is neutral (محايد) with respect to sentence 2? Sentence 1: {s1}, Sentence 2: {s2}'
        prompt_off = 'Is this text offensive (مسيء)? Text: {context}. Possible answers: yes (نعم), no (لا)'
        prompt_sarcasm = 'Is this text ironic (ساخر)? Text: {context}. Possible answers: yes (نعم), no (لا)'
        prompt_sentiment = 'What is the sentiment of the text? Text: {context}. Possible answers: negative (سلبي), neutral (محايد), positive (إيجابي)'
    elif language == 'En':
        prompt_stance = 'How would you describe the stance used in this text regarding: {target}? Text: {context}. Possible answers: neutral, against, in favor'
        prompt_nli= 'Does sentence 1 entail, contradict, or is neutral with respect to sentence 2? Sentence 1: {s1}, Sentence 2: {s2}'
        prompt_off = 'Is this text offensive? Text: {context}. Possible answers: yes, no'
        prompt_sarcasm = 'Is this text ironic? Text: {context}. Possible answers: yes, no'
        prompt_sentiment = 'What is the sentiment of the text? Text: {context}. Possible answers: negative, neutral, positive'

    
    if task == 'stance':
        prompt = prompt_stance
    elif task == 'NLI':
        prompt = prompt_nli
    elif task == 'OFF':
        prompt = prompt_off
    elif task == 'sarcasm':
        prompt = prompt_sarcasm
    elif task == 'sentiment':
        prompt = prompt_sentiment

    
    new_data = {'prompt': [], 'response': []}

    for index, row in data.iterrows():

        context = row['context']
        response = row['response']

        # Check what language for response
        if language == 'En':
            # Create a mapping dictionary
            label_map = {
                'مؤيد': 'in favor',
                'محايد': 'neutral',
                'معارض': 'against',
                'نعم': 'yes',
                'لا': 'no',
                'إيجابي': 'positive',
                'سلبي': 'negative',
                'يوافق': 'entail',
                'يناقض': 'contradict'
                }
            #Map the response column from Arabic to English
            response = label_map[response]
        
        # Check what task 
        if task == 'stance':
            target = row['target']
            prompt_text = prompt.format(target=target, context=context)
        elif task == 'NLI':
            s1 = row['s1']
            s2 = row['s2']
            prompt_text = prompt.format(s1=s1, s2=s2)    
        else:
            prompt_text = prompt.format(context=context)
        
        new_data['prompt'].append(prompt_text)
        new_data['response'].append(response)
    
    return pd.DataFrame(new_data)


#------------------------------------------------------------------

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

#file_paths_2 = ['Mawqif_stance_test.json', 'Mawqif_stance_train.json']

train_dfs = []
val_dfs = []
# Loop through each file path
for i, file in enumerate(file_paths):
    # Read the JSON file into a pandas dataframe
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    print(f"Processing file {i+1}: {file} with size: {df.shape}")

    ## Apply clean_arabic_text
    if file in files_NLI:
        # Clean the text columns
        df['s1'] = df['s1'].apply(clean_arabic_text)
        df['s2'] = df['s2'].apply(clean_arabic_text)
        # Concatnate
        df['context'] = df['s1'] + "separator" + df['s2']
    else:
        df['context'] = df['context'].apply(clean_arabic_text)
  
    ## Apply prompting
    if file in files_stance:
        df_prompt = prompting(df, "stance")
    if file in files_NLI:
        df_prompt = prompting(df, "NLI")
    if file in files_OFF:
        df_prompt = prompting(df, "OFF")
    if file in files_sarcasm:
        df_prompt = prompting(df, "sarcasm")
    if file in files_sentiment:
        df_prompt = prompting(df, "sentiment")

    # Add dataset name to each row
    df_prompt['dataset'] = file.split('.')[0] #split to remove .json

    # Add the cleaned dataframe to the list of dataframes
    if "train" in file or "traindev" in file:
        train_dfs.append(df_prompt)
    elif "test" in file:
        val_dfs.append(df_prompt)

## Merging 
#------------------------------------------------------------------
# Concatenate the dataframes into a single dataframe before calling to_json on it.
train_d = pd.concat(train_dfs, ignore_index=True)
val_d = pd.concat(val_dfs, ignore_index=True)

# Save the cleaned dataframe to a new JSON file
new_dir_path = '/home/nora_alturayeif/STANCE/My_T/data/Prompted_data_EN'
os.chdir(new_dir_path)

train_d.to_json("train.json", orient='records', lines=True, force_ascii=False)
print("Size of training data after merging:",train_d.shape)

val_d.to_json("test.json", orient='records', lines=True, force_ascii=False)
print("Size of validation data after merging:",val_d.shape)
