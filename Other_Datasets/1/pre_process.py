import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')

dataset_path = 'go_emotions_dataset.csv'
df = pd.read_csv(dataset_path)

df['text'] = df['text'].str.lower().astype(str).dropna()
df['text'] = df['text'].str.replace('[^a-z\s]', '', regex=True)
df = df[~df['example_very_unclear']]
df = df.drop(['example_very_unclear'], axis=1)

def lemmatize(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

df['text'] = df['text'].apply(lemmatize)

print(df.info())

df.to_csv('go_emotions_pre_processed.csv', index=False)