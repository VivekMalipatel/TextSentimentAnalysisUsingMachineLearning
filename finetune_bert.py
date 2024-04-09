import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig, DataCollatorWithPadding
from torch import cuda
import ast
import warnings
warnings.filterwarnings('ignore')

class Config:
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 4
    EPOCHS = 1
    LEARNING_RATE = 1e-5
    BERT_PATH = 'bert-base-multilingual-cased'
    FILE_PATH = 'Converted_Dataset.csv'
    MODEL_FOLDER = "bert_finetuned"
    MODEL_PATH = 'bert_finetuned/pytorch_model.bin'
    VOCAB_PATH = 'bert_finetuned/vocab.txt'
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
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        body = str(self.data.text[index])
        body = " ".join(body.split())
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        inputs = self.tokenizer.encode_plus(
            body,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        inputs = data_collator(inputs)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.data.labels[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len

class BERTClassifier(torch.nn.Module):
    def __init__(self):
        super(BERTClassifier, self).__init__()
        self.l1 = BertModel.from_pretrained(Config.BERT_PATH)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 28)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids, return_dict=False)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

class Trainer:
    def __init__(self, model, training_loader, testing_loader, tokenizer):
        self.model = model
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=Config.LEARNING_RATE)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.tokenizer = tokenizer

    def calcuate_accu(self, big_idx, targets):
        n_correct = (big_idx == targets).sum().item()
        return n_correct

    def train_epoch(self, epoch):
        tr_loss, n_correct, nb_tr_steps, nb_tr_examples = 0, 0, 0, 0
        self.model.train()
        for _, data in enumerate(self.training_loader, 0):
            ids = data['ids'].to(Config.device, dtype=torch.long)
            mask = data['mask'].to(Config.device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(Config.device, dtype=torch.long)
            targets = data['targets'].to(Config.device, dtype=torch.long)

            outputs = self.model(ids, mask, token_type_ids)
            loss = self.loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += self.calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            

            if _ % 500 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples 
                print(f"Training Loss per 500 steps: {loss_step}")
                print(f"Training Accuracy per 500 steps: {accu_step}")
                print()
            

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
        epoch_loss = tr_loss / nb_tr_steps
        epoch_accu = (n_correct * 100) / nb_tr_examples
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Training Accuracy Epoch: {epoch_accu}")
        print()

    def train(self):
        loss= None
        for epoch in range(Config.EPOCHS):
            self.train_epoch(epoch)
        self.save_model()

    def save_model(self):
        config = BertConfig.from_pretrained(Config.BERT_PATH)
        config.num_labels=28
        config.architectures = "BertForForSequenceClassification"
        config.label2id = {'admiration':0, 'amusement':1, 'anger':2, 'annoyance':3, 'approval':4, 'caring':5, 'confusion':6, 'curiosity':7, 'desire':8, 'disappointment':9, 'disapproval':10, 'disgust':11, 'embarrassment':12, 'excitement':13, 'fear':14, 'gratitude':15, 'grief':16, 'joy':17, 'love':18, 'nervousness':19, 'optimism':20, 'pride':21, 'realization':22, 'relief':23, 'remorse':24, 'sadness':25, 'surprise':26, 'neutral':27}
        config.id2label = {0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval', 5: 'caring', 6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment', 10: 'disapproval', 11: 'disgust', 12: 'embarrassment', 13: 'excitement', 14: 'fear', 15: 'gratitude', 16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness', 20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief', 24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral'}
        config.save_pretrained(Config.MODEL_FOLDER)
        self.model.eval()
        torch.save({
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict()
        }, Config.MODEL_PATH, _use_new_zipfile_serialization=False)
        #self.tokenizer.save_vocabulary(Config.VOCAB_PATH)
        self.tokenizer.save_pretrained(Config.MODEL_FOLDER)
        print('Model and tokenizer have been saved.')

class Validator:
    def __init__(self, model, testing_loader):
        self.model = model
        self.testing_loader = testing_loader
        self.loss_function = torch.nn.CrossEntropyLoss()
    
    def validate(self):
        self.model.eval()
        tr_loss, n_correct, nb_tr_steps, nb_tr_examples = 0, 0, 0, 0
        with torch.no_grad():
            for _, data in enumerate(self.testing_loader, 0):
                ids = data['ids'].to(Config.device, dtype=torch.long)
                mask = data['mask'].to(Config.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(Config.device, dtype=torch.long)
                targets = data['targets'].to(Config.device, dtype=torch.long)
                outputs = self.model(ids, mask, token_type_ids).squeeze()
                loss = self.loss_function(outputs, targets)
                tr_loss += loss.item()
                big_val, big_idx = torch.max(outputs.data, dim=1)
                n_correct += (big_idx == targets).sum().item()

                nb_tr_steps += 1
                nb_tr_examples += targets.size(0)

        epoch_loss = tr_loss / nb_tr_steps
        epoch_accu = (n_correct * 100) / nb_tr_examples
        print(f"Validation Loss: {epoch_loss}")
        print(f"Validation Accuracy: {epoch_accu}%")
        return epoch_accu

if __name__ == "__main__":
    preprocessor = EmailDatasetPreprocessor(Config.FILE_PATH)
    df = preprocessor.preprocess()
    
    train_size = 0.8
    train_dataset = df.sample(frac=train_size, random_state=200).reset_index(drop=True)
    test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    
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

    '''
    validator = Validator(model, testing_loader)
    validator.validate()
    '''

