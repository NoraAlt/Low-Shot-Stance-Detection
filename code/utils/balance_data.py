import json
import random

tasks = {
    'Stance': ['AraCovidVac_test'],
    'Stance_mawqif': ['Mawqif_stance_test'],
    'NLI+Claim': ['XGLUE-xnli_test'],
    'Claim': ['AraStance_test','Headlines_test'],
    'Offensive': ['OSACT2022_test',  'OSACT2020_test'],
    'Sarcasm': ['ArSarcasm_sarcastic_test','Mawqif_sarcasm_test','iSarcasmEval_test'],
    'Sentiment': ['AJGT_test','ASTD_test', 'MASC_test', 'UMSAB_test','Mawqif_sentiment_test', 'ArSarcasm_sentiment_test']
}



num_samples_per_task = 3000

# Create a dictionary to store samples for each task
task_samples = {task: [] for task in tasks}

# Load the input JSONL data
with open('/home/nora_alturayeif/STANCE/My_T/data/Prompted_data_EN/test.json', 'r') as f:
    for line in f:
        sample = json.loads(line)
        dataset = sample['dataset']

        # Check which task this dataset belongs to
        for task, datasets in tasks.items():
            if dataset in datasets:
                task_samples[task].append(sample)
                break  # No need to continue checking other tasks

# Shuffle the samples within each task
for task, samples in task_samples.items():
    # Print the number of samples per task before extraction
    print(f"Number of samples in '{task}': {len(samples)}")
    random.shuffle(samples)

# Extract a certain number of samples from each task
extracted_samples = []
for task, samples in task_samples.items():
    if len(samples) >= num_samples_per_task:
        extracted_samples.extend(samples[:num_samples_per_task])
    else:
        extracted_samples.extend(samples)

print(f"Number of extracted samples combined: {len(extracted_samples)}")

# Write the extracted samples to a new JSONL file

with open('Test_balanced.jsonl', 'w', encoding='utf-8') as f:
    for sample in extracted_samples:
        json.dump(sample, f, ensure_ascii=False)
        f.write('\n')


