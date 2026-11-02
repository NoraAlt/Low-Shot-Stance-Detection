import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Read the CSV file
data_name = "MASC"
file_name = "MASC_All.csv"
data = pd.read_csv(file_name) 

# Encode the response column to numerical values
le = LabelEncoder()
data['response'] = le.fit_transform(data['response'])

print("Original Train set size:", data.shape)

# Split the data into train and val sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['response'])
# Split the data into train, validation, and test sets
#train_data, val_test_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['response'])
#val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42, stratify=val_test_data['response'])

# Check the size of the train and val sets
print("Train set size:", train_data.shape)
print("Val set size:", test_data.shape)
#print("Validation set size:", val_data.shape)
#print("Test set size:", test_data.shape)


# Save the train and test sets to CSV files
train_data.to_csv(data_name + "_train.csv", index=False, encoding='utf-8-sig')
test_data.to_csv(data_name + "_val.csv", index=False, encoding='utf-8-sig')
#test_data.to_csv(data_name + "_test.csv", index=False, encoding='utf-8-sig')
