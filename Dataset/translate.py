import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm
tqdm.pandas()

translator = GoogleTranslator(source='en', target='es')

def translate_text(text):
    try:
        translated_text = translator.translate(text)
        return str(translated_text)
    except Exception as e:
        print("Error during translation:", e)
        return "NAAN" + text


data_path = 'Dataset/Testing_dataset/pre_processed_text_test_partition.csv'
data = pd.read_csv(data_path)

print("Translating text to Spanish...")
data['text'] = data['text'].progress_apply(translate_text)

print("Saving translated data...")
data.to_csv('Dataset/Testing_dataset/pre_processed_text_test_partition_es_translated.csv', index=False)