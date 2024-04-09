import pandas as pd

data = pd.read_csv('go_emotions_dataset.csv')

# Dropping non-emotion columns to focus on the emotion labels
emotion_data = data.drop(['id', 'text', 'example_very_unclear'], axis=1)

# Summing up the values in each emotion label column
emotion_counts = emotion_data.sum()

print(emotion_counts)