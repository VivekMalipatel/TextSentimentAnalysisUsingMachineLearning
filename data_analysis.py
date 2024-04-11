import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def load_data():
    # Load the pre-processed data
    data = pd.read_csv('pre_processed_text.csv')
    return data

def plot_label_distribution(data):
    # Plotting the distribution of labels
    plt.figure(figsize=(10, 6))
    sns.countplot(x='label', data=data)
    plt.title('Distribution of Labels')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

def plot_word_count_distribution(data):
    # Adding a new column for word count
    data['word_count'] = data['text'].apply(lambda x: len(x.split()))
    
    # Plotting the distribution of word counts
    plt.figure(figsize=(10, 6))
    sns.histplot(data['word_count'], bins=30, kde=True)
    plt.title('Distribution of Word Counts in Texts')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def generate_word_cloud(data):
    # Concatenate all texts to create a word cloud
    text = " ".join(review for review in data.text)
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def main():
    # Load the dataset
    data = load_data()
    
    # Display basic info and description
    print("Data Overview:")
    print(data.info())
    print(data.describe())

    # Plots
    plot_label_distribution(data)
    plot_word_count_distribution(data)
    generate_word_cloud(data)

if __name__ == '__main__':
    main()
