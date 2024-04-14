import pandas as pd

class EmotionCompressor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.emotion_mapping = {
            'empty': 0, 'sadness': 0, 'boredom': 0,
            'happiness': 1, 'fun': 1, 'enthusiasm': 1, 'relief': 1,
            'love': 2,
            'anger': 3, 'hate': 3,
            'worry': 4,
            'surprise': 5,
        }

    def process_file(self):
        data = pd.read_csv(self.input_file)
        data = data[data['label'] != 'neutral']
        data['label'] = data['label'].map(self.emotion_mapping)
        data['label'].fillna('Unknown', inplace=True)
        data.to_csv(self.output_file, index=False)

if __name__ == '__main__':
    compressor = EmotionCompressor('Dataset/Testing_dataset/pre_processed_text_emotion.csv', 'Dataset/Testing_dataset/pre_processed_text_emotion_compressed.csv')
    compressor.process_file()