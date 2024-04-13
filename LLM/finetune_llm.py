import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, accuracy_score

class Config:

    #Paths
    INPUT_DATA_PATH = 'pre_processed_text.csv'
    BASE_MODEL_PATH = 'LLM/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
    MODEL_TRAINING_LOGS_PATH = 'LLM/LLM_training_logs'
    FINETUNED_MODEL_SAVE_PATH = 'LLM/Finetuned_LLM_model_files'

    SEED_GLOBAL = 42
    TRAIN_SIZE = 0.8

    HYPOTHESIS_LABELS = {
        0 : 'The text portrays a sadness sentiment.',
        1 : 'The text portrays a joy sentiment.',
        2 : 'The text portrays a love sentiment.',
        3 : 'The text portrays a anger sentiment.',
        4 : 'The text portrays a fear sentiment.',
        5 : 'The text portrays a surprise sentiment.',
    }

    # Model parameters
    DEVICE = 'cuda' if torch.cuda.is_available() else torch.device("mps")
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 64
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    GRADIENT_ACCUMULATION_STEPS = 1
    WARMUP_RATIO = 0.06
    WEIGHT_DECAY = 0.01
    FP16_BOOL = False

class PrepareDataset:

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.train_dataset = None

    def fit_hypothesis(self, text):
        return 'The text is - "' + text + '" - end of text'
    
    def format_nli_trainset(self, data, hypothesis_labels, random_seed=Config.SEED_GLOBAL):
        print('Formatting NLI trainset...')
        print(f"Length of Train data before formatting step: {len(data)}.")
        length_original_data = len(data)
        df_lst = []
        for label, hypothesis in hypothesis_labels.items():
            ## entailment
            df_step = data[data.label == label].copy(deep=True)
            df_step["hypothesis"] = [hypothesis] * len(df_step)
            df_step["labels"] = [0] * len(df_step)
            ## not_entailment
            df_step_not_entail = data[data.label != label].copy(deep=True)
            df_step_not_entail = df_step_not_entail.sample(n=min(len(df_step), len(df_step_not_entail)), random_state=random_seed)
            df_step_not_entail["hypothesis"] = [hypothesis] * len(df_step_not_entail)
            df_step_not_entail["labels"] = [1] * len(df_step_not_entail)
            # append
            df_lst.append(pd.concat([df_step, df_step_not_entail]))
        data = pd.concat(df_lst)
        # shuffle
        data = data.sample(frac=1, random_state=random_seed)
        print(f"After adding not_entailment training examples, the training data was augmented to {len(data)} texts.")
        return data.copy(deep=True)
    
    def format_nli_testset(self, data, hypothesis_labels):

        print('Formatting NLI testset...')
        ## explode test dataset for N hypotheses
        hypothesis_lst = [value for key, value in hypothesis_labels.items()]
        print("Number of hypotheses/classes:", len(hypothesis_lst))
        # label lists with 0 at alphabetical position of their true hypo, 1 for not-true hypos
        label_text_label_dic_explode = {}
        for key, value in hypothesis_labels.items():
            label_lst = [0 if value == hypo else 1 for hypo in hypothesis_lst]
            label_text_label_dic_explode[key] = label_lst
        data["labels"] = data.label.map(label_text_label_dic_explode)
        data["hypothesis"] = [hypothesis_lst] * len(data)
        print(f"Original test set size: {len(data)}")
        data = data.explode(["hypothesis", "labels"])
        print(f"Test set size for NLI classification: {len(data)}")
        return data.copy(deep=True)
    
    def convert_to_dataset(self, train_dataset, test_dataset):
        columns_to_keep = ["text", "hypothesis", "labels"]
        return datasets.DatasetDict({"train" : datasets.Dataset.from_pandas(train_dataset[columns_to_keep]),
                                    "test" : datasets.Dataset.from_pandas(test_dataset[columns_to_keep])})
    
    def tokenize_function(self, dataset):
        return self.tokenizer(dataset["text"], dataset["hypothesis"], padding=True, truncation = True, max_length = Config.MAX_LEN, return_tensors="pt")
    
    def prepare_dataset(self):

        print('\nPreparing dataset...')
        print('Fitting hypothesis...')
        self.data['text'] = self.data['text'].progress_apply(self.fit_hypothesis)
        train_dataset = self.data.sample(frac=Config.TRAIN_SIZE, random_state=Config.SEED_GLOBAL).reset_index(drop=True)
        test_dataset = self.data.drop(train_dataset.index).reset_index(drop=True)
        print('Train dataset shape:', train_dataset.shape)
        print('Test dataset shape:', test_dataset.shape)
        train_dataset = self.format_nli_trainset(train_dataset, Config.HYPOTHESIS_LABELS)
        test_dataset = self.format_nli_testset(test_dataset, Config.HYPOTHESIS_LABELS)
        dataset = self.convert_to_dataset(train_dataset, test_dataset)
        print('Tokenizing Train and Test datasets...')
        dataset = dataset.map(self.tokenize_function, batched=True)
        print('Dataset preparation done.')
        return dataset

class Train:
    
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer

        self.train_args = TrainingArguments(
            output_dir=Config.MODEL_TRAINING_LOGS_PATH,
            logging_dir=f'{Config.MODEL_TRAINING_LOGS_PATH}/logs',
            learning_rate=Config.LEARNING_RATE,
            lr_scheduler_type= "linear",
            group_by_length=False, 
            gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
            per_device_train_batch_size=Config.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=Config.VALID_BATCH_SIZE,
            num_train_epochs=Config.EPOCHS,
            warmup_ratio=Config.WARMUP_RATIO,
            weight_decay=Config.WEIGHT_DECAY,
            fp16=Config.FP16_BOOL,
            fp16_full_eval=Config.FP16_BOOL,
            seed = Config.SEED_GLOBAL,
            #load_best_model_at_end= True,
            #metric_for_best_model="accuracy",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit = 1,
            report_to="all"
        )

        self.trainer = Trainer(
            model = self.model,
            tokenizer = self.tokenizer,
            args = self.train_args,
            train_dataset = self.dataset["train"],
            eval_dataset = self.dataset["test"],
            compute_metrics = lambda eval_pred : Evaluate().compute_metrics_nli_binary(eval_pred)
        )
    
    def save_model(self):
        print("Saving model...")
        self.trainer.save_model(Config.FINETUNED_MODEL_SAVE_PATH)
        print(f"Model saved at {Config.FINETUNED_MODEL_SAVE_PATH}")

    def train_model(self):
        self.trainer.train()
        self.save_model()

class Evaluate:
    
    def __init__(self):
        self.labels_set = Config.HYPOTHESIS_LABELS.keys()
    
    def compute_metrics_nli_binary(self, eval_pred):

        def chunks(list, length):
            for i in range(0, len(list), length):
                yield list[i:i + length]

        predictions, labels = eval_pred
        softmax = torch.nn.Softmax(dim=1)
        predictions_chunks_list = list(chunks(predictions,len(self.labels_set)))
        hypo_position_highest_prob = []
        for i, chunk in enumerate(predictions_chunks_list):
            hypo_position_highest_prob.append(np.argmax(np.array(chunk)[:, 0]))
        label_chunks_list = list(chunks(labels, len(self.labels_set)))
        label_position_gold = []
        for chunk in label_chunks_list:
            label_position_gold.append(np.argmin(chunk))
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='macro')
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='micro')
        acc_balanced = balanced_accuracy_score(label_position_gold, hypo_position_highest_prob)
        acc_not_balanced = accuracy_score(label_position_gold, hypo_position_highest_prob)
        metrics = {
            'accuracy': acc_not_balanced,
            'f1_macro': f1_macro,
            'accuracy_balanced': acc_balanced,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
        }
        return metrics


if __name__ == "__main__":

    data = pd.read_csv(Config.INPUT_DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_PATH, model_max_length = Config.MAX_LEN)
    model = AutoModelForSequenceClassification.from_pretrained(Config.BASE_MODEL_PATH)
    model.to(Config.DEVICE)
    
    prepare_dataset = PrepareDataset(data, tokenizer)
    dataset = prepare_dataset.prepare_dataset()

    print("The overall structure of the prepared dataset:")
    print(dataset)
    print("sample row of the training dataset:")
    print(dataset['train'][:1])
    print("sample row of the testing dataset:")
    print(dataset['test'][:1])

    print("The model training is starting...")
    trainer = Train(model, tokenizer, dataset)
    trainer.train_model()
    print("Model training is done.")