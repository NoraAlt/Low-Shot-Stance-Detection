import json
import random
import os


def extract_n_way_k_shot_instances(data, target_dataset_name, n_way, k_shot, filter_prompt):
    n_way_k_shot_data = []


    # Filter the data for the specified target dataset and prompt condition if provided
    if filter_prompt: #if hide target
        target_dataset_entries = [entry for entry in data if filter_prompt in entry['prompt']]
    else: # if hide task
        target_dataset_entries = [entry for entry in data if entry['dataset'] in target_dataset_name]


    # Shuffle each class individually to get balanced instances
    class_groups = {'in favor': [], 'against': [], 'neutral': []}
    for entry in target_dataset_entries:
        response = entry['response']
        class_groups[response].append(entry)

    # Calculate the number of samples per class based on the desired ratio
    num_samples_favor_against = (n_way * k_shot * 2) // 5
    num_samples_neutral = (n_way * k_shot) // 5


    # Sample instances for each class based on the calculated sample numbers
    if filter_prompt:
        for target in range(n_way):
            for response in class_groups.keys():
                num_samples = num_samples_favor_against if response in ['in favor', 'against'] else num_samples_neutral
                sampled_entries = random.sample(class_groups[response], num_samples)
                n_way_k_shot_data.extend(sampled_entries)
    else:
        for target in targets:
            for response in class_groups.keys():
                num_samples = num_samples_favor_against if response in ['in favor', 'against'] else num_samples_neutral
                target_filtered_entries = [entry for entry in class_groups[response] if target in entry['prompt']]
                sampled_entries = random.sample(target_filtered_entries, num_samples)
                n_way_k_shot_data.extend(sampled_entries)


    return n_way_k_shot_data


#########################################


def main(data_file_path, target_dataset_name, filter_prompt, n_way, k_shot, new_dir_path, output_file_name):
    # Read data from the specified directory
    with open(data_file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    # Extract N-way K-shot instances from the target dataset and filtered prompt
    n_way_k_shot_data = extract_n_way_k_shot_instances(data, target_dataset_name, n_way, k_shot, filter_prompt)
    
    # Filter the original data and remove the target dataset and filtered prompt
    if filter_prompt:
        filtered_data = [entry for entry in data if not (entry['dataset'] in target_dataset_name and filter_prompt in entry['prompt'])]
    else:
        filtered_data = [entry for entry in data if not (entry['dataset'] in target_dataset_name)]

    # Append the extracted N-way K-shot data to the filtered data
    updated_data = filtered_data + n_way_k_shot_data

    # Print the number of entries in each data set
    print("Number of entries in original data:", len(data))
    print("Number of entries in n_way_k_shot_data:", len(n_way_k_shot_data))
    print("Number of entries in filtered_data:", len(filtered_data))
    print("Number of entries in updated_data:", len(updated_data))

    # Save the updated data to a new JSON file
    os.chdir(new_dir_path)

    with open(output_file_name, 'w', encoding='utf-8') as outfile:
        for entry in updated_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')

    print(f"Data saved successfully to {output_file_name}")



if __name__ == "__main__":

    n_way = 3 # Number of classes
    k_shot = 20  # Number of instances per class

    # Set the target dataset name and the filter prompt
    #target_dataset_name = "Mawqif_stance_train"
    target_dataset_name = ["Mawqif_stance_train", "AraCovidVac_traindev"]

    targets = ["regarding: لقاح كورونا?", "regarding: تمكين المرأة?" , "regarding: التحول الرقمي?"]

    filter_prompt = "regarding: لقاح كورونا?" #"regarding: التحول الرقمي?" #"regarding: تمكين المرأة?" # # #None #

    # Define the file paths
    data_file_path = "/home/nora_alturayeif/STANCE/My_T/data/Prompted_data_EN/train.json"
    new_dir_path = '/home/nora_alturayeif/STANCE/My_T/data/Low-shot'

    # Define the output file name
    output_file_name = "train_"+ str(n_way)+ "N_"+ str(k_shot)+ "K_Covid.jsonl" ### CHANGE THE NAME ####

    # Execute the main function
    main(data_file_path, target_dataset_name, filter_prompt, n_way, k_shot, new_dir_path, output_file_name)

