import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    # Load the pre-processed data
    data = pd.read_csv('Dataset/Testing_dataset/text_emotion.csv')
    text_column = 'content'
    label_column = 'sentiment'
    return data, text_column, label_column

def plot_label_distribution(data, label_column):
    # Plotting the distribution of labels
    plt.figure(figsize=(10, 6))
    sns.countplot(x=label_column, data=data)
    plt.title('Distribution of Labels')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

def plot_word_count_distribution(data, text_column):
    # Adding a new column for word count
    data['word_count'] = data[text_column].apply(lambda x: len(x.split()))
    
    # Plotting the distribution of word counts
    plt.figure(figsize=(10, 6))
    sns.histplot(data['word_count'], bins=30, kde=True)
    plt.title('Distribution of Word Counts in Texts')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def main():
    # Load the dataset
    data, text_column, label_column = load_data()
    
    # Display basic info and description
    print("Data Overview:")
    print(data.info())
    print(data.describe())

    # Plots
    plot_label_distribution(data, label_column)
    plot_word_count_distribution(data, text_column)

if __name__ == '__main__':
    main()
