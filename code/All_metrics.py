import json
import csv
from sklearn.metrics import precision_recall_fscore_support


# Define the possible classes
classes = ["in favor", "against", "neutral"]

# Define label mapping
label_map = {
    'in favor': 0,
    'neutral': 1,
    'against': 2
}

# Initialize dictionaries to store data for each target
target_scores = {"women": [], "digital": [], "covid": []}
all_references = []
all_predictions = []

# Data to evaluate: !!!! CHANGE HERE !!!!
data_name = '/home/nora_alturayeif/STANCE/My_T/output/zero_All/eval/test_Mawqif_generated_predictions.json'

# Read the JSON lines format file
with open(data_name, 'r', encoding='utf-8') as file:
    data_list = [json.loads(line) for line in file]

for data in data_list:
    response = data["response"]
    prediction = data["predictions"]

    # Map the response and prediction to label indices
    response_mapped = label_map[response]
    prediction_mapped = label_map[prediction]

    all_references.append(response_mapped)
    all_predictions.append(prediction_mapped)

    # Extract target from prompt
    if "تمكين المرأة" in data["prompt"]:
        target = "women"
    elif "التحول الرقمي" in data["prompt"]:
        target = "digital"
    elif "لقاح كورونا" in data["prompt"]:
        target = "covid"
    else:
        target = None

    if target:
        # Append the response and prediction to target-specific lists
        target_scores[target].append((response_mapped, prediction_mapped))

# Calculate precision, recall, and F1 score for all references and predictions
precision, recall, f1, _ = precision_recall_fscore_support(all_references, all_predictions, labels=[0, 1, 2])

# Calculate the customized F1 score as the average of the F1 score for favor and against classes
custom_f1_overall = (f1[0] + f1[2]) / 2


# Write the scores to a CSV file
with open('All_targets_scores.csv', 'w', newline='') as csvfile:
    fieldnames = ['Target', 'Custom_F1']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Write target-specific scores to the CSV file
    for target, target_data in target_scores.items():
        target_references, target_predictions = zip(*target_data)
        target_precision, target_recall, target_f1, _ = precision_recall_fscore_support(target_references, target_predictions, labels=[0, 1, 2])
        target_custom_f1 = (target_f1[0] + target_f1[2]) / 2
        writer.writerow({'Target': target, 'Custom_F1': target_custom_f1})

    # Write overall score to the CSV file
    writer.writerow({'Target': 'Overall', 'Custom_F1': custom_f1_overall})

print("Saved to All_targets_scores.csv")
