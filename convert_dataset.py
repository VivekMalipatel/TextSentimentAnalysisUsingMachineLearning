import pandas as pd

dataset_path = 'go_emotions_pre_processed.csv'
data = pd.read_csv(dataset_path)

emotion_labels = data.columns[2:]

# Function to concatenate emotion labels present in each text (where value is 1)
def concatenate_labels(row):
    labels = [label for label in emotion_labels if row[label] == 1]
    return ", ".join(labels) if labels else "neutral"  # Default to "neutral" if no labels are present

# Apply the function to each row
data['labels'] = data.apply(concatenate_labels, axis=1)

# Select only the text and concatenated labels columns for the final output
final_data = data[['text', 'labels']]

final_data.to_csv('Converted_Dataset.csv')