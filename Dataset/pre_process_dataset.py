import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from textblob import TextBlob
import re
from tqdm import tqdm
tqdm.pandas()

# Download necessary NLTK data
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

def expand_contractions(text):
    contractions_dict = {
        "ive": "I have", "im": "I am", "youre": "you are", "were": "we are",
        "theyre": "they are", "cant": "cannot", "couldnt": "could not",
        "dont": "do not", "doesnt": "does not", "its": "it is", "thats": "that is",
        "theres": "there is"
    }
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
    return contractions_re.sub(lambda match: contractions_dict[match.group(0)], text)

def reduce_elongation(text):
    return re.compile(r"(.)\1{2,}").sub(r"\1\1", text)

def correct_spelling(text):
    return str(TextBlob(text).correct())

def nltk_pos_to_wordnet_pos(nltk_pos):
    pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
    return pos_dict.get(nltk_pos[0], None)

def clean_and_lemmatize(text):
    text = expand_contractions(text)
    text = reduce_elongation(text)
    text = correct_spelling(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    nltk_pos_tags = nltk.pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(word, nltk_pos_to_wordnet_pos(tag)) if nltk_pos_to_wordnet_pos(tag) else word for word, tag in nltk_pos_tags]
    return ' '.join(lemmatized_tokens)

def main():
    # Load the dataset
    data_path = 'Dataset/Testing_dataset/text_emotion.csv'
    data_save_path = 'Dataset/Testing_dataset/pre_processed_text.csv'
    data = pd.read_csv(data_path)
    text_column = 'content'
    label_column = 'sentiment'
    columns_to_keep = [text_column, label_column]
    data = data[columns_to_keep]
    print("Dataset Overview:")
    print(data.describe())
    print("First 5 rows of the dataset:")
    print(data.head())
    print("Cleaning Text :")
    data[text_column] = data[text_column].progress_apply(clean_and_lemmatize)
    print("Cleaning Done!")
    print("After Cleaning Text :")
    print(data.head())
    data.columns = ['text', 'label']
    print("Saving the cleaned data to a new CSV file...")
    data.to_csv(data_save_path,index=False)
    print("Data saved to", data_save_path)

if __name__ == '__main__':
    main()
