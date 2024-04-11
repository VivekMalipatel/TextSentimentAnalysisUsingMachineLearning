import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig, __version__
import datasets
import torch
import warnings
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, accuracy_score
warnings.filterwarnings('ignore')
import os

class Config:

    TRANSFORMER_VERSION = __version__

    BASE_MODEL_PATH = ''
    DATA_FILE_PATH = ''
    LOGS_PATH = ''
    FINETUNED_SAVE_PATH = ' '

    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 4 if "large" in BASE_MODEL_PATH else 8
    VALID_BATCH_SIZE = 64 if "large" in BASE_MODEL_PATH else 64*2
    EPOCHS = 20
    LEARNING_RATE = 9e-6 if "large" in BASE_MODEL_PATH else 2e-5
    WARMUP_RATIO = 0.06
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 4 if "large" in BASE_MODEL_PATH else 1

    FP16_BOOL = True if torch.cuda.is_available() else False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_size = 0.75
    SEED_GLOBAL = 42
    np.random.seed(SEED_GLOBAL)
    torch.manual_seed(SEED_GLOBAL)

    hypothesis_label_dic = {

   }