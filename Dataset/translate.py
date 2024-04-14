import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm
tqdm.pandas()

translator = GoogleTranslator(source='en', target='es')

def translate_text(text):
    try:
        # Translate the text to Spanish using DeepL (as an example)
        translated_text = translator.translate(text)
        print("Original text:", text)
        print("Translated text:", translated_text)
        return str(translated_text)
    except Exception as e:
        # In case of any error, return the original text
        print("Error during translation:", e)
        return "NAAN" + text


data_path = 'Dataset/Testing_dataset/pre_processed_text_test_partition.csv'
data = pd.read_csv(data_path)

data = data.sample(5)

print("Translating text to Spanish...")
data['text'] = data['text'].progress_apply(translate_text)

print("Saving translated data...")
data.to_csv('Dataset/Testing_dataset/pre_processed_text_test_partition_es_translated.csv', index=False)