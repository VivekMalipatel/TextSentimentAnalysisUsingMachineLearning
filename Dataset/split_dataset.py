import pandas as pd
from sklearn.model_selection import train_test_split

data_path = 'Dataset/Training_dataset/pre_processed_text.csv'
data = pd.read_csv(data_path)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_data.to_csv('Dataset/Training_dataset/pre_processed_text_train_partition.csv', index=False)
test_data.to_csv('Dataset/Testing_dataset/pre_processed_text_test_partition.csv', index=False)