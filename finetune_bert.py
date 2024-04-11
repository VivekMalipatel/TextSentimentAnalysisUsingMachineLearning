import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig, DataCollatorWithPadding
from torch import cuda
import ast
import numpy as np
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

class Config:
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 32
    EPOCHS = 3
    LEARNING_RATE = 1e-5
    BERT_PATH = 'bert-base-multilingual-cased'
    TEMP_PATH = 'temp_finetuned'
    FILE_PATH = 'Converted_Dataset.csv'
    MODEL_FOLDER = "bert_finetuned"
    MODEL_NAME = 'pytorch_model.bin'
    VOCAB_NAME = 'vocab.txt'
    device = 'cuda:5' if cuda.is_available() else 'cpu'

class EmailDatasetPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.encode_dict = {'admiration':0, 'amusement':1, 'anger':2, 'annoyance':3, 'approval':4, 'caring':5, 'confusion':6, 'curiosity':7, 'desire':8, 'disappointment':9, 'disapproval':10, 'disgust':11, 'embarrassment':12, 'excitement':13, 'fear':14, 'gratitude':15, 'grief':16, 'joy':17, 'love':18, 'nervousness':19, 'optimism':20, 'pride':21, 'realization':22, 'relief':23, 'remorse':24, 'sadness':25, 'surprise':26, 'neutral':27}
        
    def encode_cat(self, x):
        if x not in self.encode_dict.keys():
            self.encode_dict[x] = len(self.encode_dict)
        return self.encode_dict[x]
    
    def preprocess(self):
        df = pd.read_csv(self.file_path, encoding='utf-8', converters={'labels': ast.literal_eval})
        df = df[['text', 'labels']]
        print(df.head())
        #df['ENCODE_CAT'] = df['Label'].apply(lambda x: self.encode_cat(x))
        return df

class TriageDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        text = str(self.data.text[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.data.labels[index], dtype=torch.float)
        } 
    
    def __len__(self):
        return len(self.data.text)

class BERTClassifier(torch.nn.Module):
    def __init__(self):
        super(BERTClassifier, self).__init__()
        self.l1 = BertModel.from_pretrained(Config.BERT_PATH)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 28)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, output_1= self.l1(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

class Trainer:
    def __init__(self, model, training_loader, testing_loader, tokenizer):
        self.model = model
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=Config.LEARNING_RATE)
        self.loss_function = torch.nn.BCEWithLogitsLoss()
        self.tokenizer = tokenizer

    def calcuate_accu(self, big_idx, targets):
        n_correct = (big_idx == targets).sum().item()
        return n_correct

    def train_epoch(self, epoch):
        self.model.train()
        for _, data in enumerate(self.training_loader, 0):
            ids = data['ids'].to(Config.device, dtype=torch.long)
            mask = data['mask'].to(Config.device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(Config.device, dtype=torch.long)
            targets = data['targets'].to(Config.device, dtype=torch.float)
            
            outputs = self.model(ids, mask, token_type_ids)
            self.optimizer.zero_grad()
            loss = self.loss_function(outputs, targets)

            if _ % 5000 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
                self.save_model(True)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self):
        loss= None
        for epoch in range(Config.EPOCHS):
            self.train_epoch(epoch)
        self.save_model()

    def save_model(self, temp = False):
        
        save_path = Config.MODEL_FOLDER
        if temp:
            save_path = Config.TEMP_PATH

        config = BertConfig.from_pretrained(Config.BERT_PATH)
        config.num_labels=28
        config.architectures = "BertForForSequenceClassification"
        config.label2id = {'admiration':0, 'amusement':1, 'anger':2, 'annoyance':3, 'approval':4, 'caring':5, 'confusion':6, 'curiosity':7, 'desire':8, 'disappointment':9, 'disapproval':10, 'disgust':11, 'embarrassment':12, 'excitement':13, 'fear':14, 'gratitude':15, 'grief':16, 'joy':17, 'love':18, 'nervousness':19, 'optimism':20, 'pride':21, 'realization':22, 'relief':23, 'remorse':24, 'sadness':25, 'surprise':26, 'neutral':27}
        config.id2label = {0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval', 5: 'caring', 6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment', 10: 'disapproval', 11: 'disgust', 12: 'embarrassment', 13: 'excitement', 14: 'fear', 15: 'gratitude', 16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness', 20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief', 24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral'}
        config.save_pretrained(save_path)
        self.model.eval()
        torch.save({
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict()
        }, save_path+'/'+Config.MODEL_NAME, _use_new_zipfile_serialization=False)
        #self.tokenizer.save_vocabulary(Config.VOCAB_PATH)
        self.tokenizer.save_pretrained(save_path)
        if not temp:
            print('Model and tokenizer have been saved.')

class Validator:
    def __init__(self, model, testing_loader):
        self.model = model
        self.testing_loader = testing_loader
        self.loss_function = torch.nn.BCEWithLogitsLoss()
    
    def validation(self, epoch):
        self.model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for _, data in enumerate(self.testing_loader, 0):
                ids = data['ids'].to(Config.device, dtype=torch.long)
                mask = data['mask'].to(Config.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(Config.device, dtype=torch.long)
                targets = data['targets'].to(Config.device, dtype=torch.float)
                outputs = model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets

    def validate(self):
        for epoch in range(Config.EPOCHS):
            outputs, targets = self.validation(epoch)
            outputs = np.array(outputs) >= 0.5
            accuracy = metrics.accuracy_score(targets, outputs)
            f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
            f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
            print(f"Accuracy Score = {accuracy}")
            print(f"F1 Score (Micro) = {f1_score_micro}")
            print(f"F1 Score (Macro) = {f1_score_macro}")

if __name__ == "__main__":
    preprocessor = EmailDatasetPreprocessor(Config.FILE_PATH)
    df = preprocessor.preprocess()
    
    train_size = 0.8
    train_dataset = df.sample(frac=train_size, random_state=200).reset_index(drop=True)
    test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    
    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    tokenizer = BertTokenizer.from_pretrained(Config.BERT_PATH)
    
    training_set = TriageDataset(train_dataset, tokenizer, Config.MAX_LEN)
    testing_set = TriageDataset(test_dataset, tokenizer, Config.MAX_LEN)

    train_params = {'batch_size': Config.TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    test_params = {'batch_size': Config.VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = BERTClassifier().to(Config.device)

    trainer = Trainer(model, training_loader, testing_loader, tokenizer)

    trainer.train()

    validator = Validator(model, testing_loader)
    validator.validate()