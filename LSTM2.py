import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter
from torchtext.vocab import vocab
import os

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        out = self.fc(hidden[-1])
        return out

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    init_process_group(backend='gloo', rank=rank, world_size=world_size)

def evaluate(model, loader, criterion, rank):
    model.eval()
    total_loss, total_accuracy, total_precision, total_recall, total_f1 = 0, 0, 0, 0, 0
    num_batches = len(loader)
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(rank), labels.to(rank)
            outputs = model(texts, [len(x) for x in texts])
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            accuracy = accuracy_score(labels.cpu(), predictions.cpu())
            total_accuracy += accuracy
            precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), predictions.cpu(), average='macro')
            total_precision += precision
            total_recall += recall
            total_f1 += f1
    return total_loss / num_batches, total_accuracy / num_batches, total_precision / num_batches, total_recall / num_batches, total_f1 / num_batches

def main(rank, world_size):
    ddp_setup(rank, world_size)

    # Load the pre-processed dataset
    data_path = 'pre_processed_text.csv'
    data = pd.read_csv(data_path)
    
    # Extract features and labels
    X = data['text'].astype(str)
    y = data['label']

    # Create a vocabulary
    token_counts = Counter(word for sentence in X for word in sentence.split())
    my_vocab = vocab(token_counts, min_freq=2)
    vocab_size = len(my_vocab)

    # Encode text and labels
    encoded_texts = [[my_vocab[word] for word in text.split() if word in my_vocab] for text in X]
    encoded_labels = y

    # Create tensors
    text_tensor = [torch.tensor(x) for x in encoded_texts]
    label_tensor = torch.tensor(encoded_labels)

    # Padding sequences to create uniform length
    padded_texts = pad_sequence(text_tensor, batch_first=True)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(padded_texts, label_tensor, test_size=0.2, random_state=42)

    # Describe X_train and X_test
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("X_train sample:", X_train[0])
    print("X_test sample:", X_test[0])

    # Create data loaders
    train_sampler = DistributedSampler(TensorDataset(X_train, y_train), num_replicas=world_size, rank=rank)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=False, sampler=train_sampler)
    test_sampler = DistributedSampler(TensorDataset(X_test, y_test), num_replicas=world_size, rank=rank)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=4096, shuffle=False, sampler=test_sampler)

    # Model parameters
    embedding_dim = 100
    hidden_dim = 256
    output_dim = len(set(y))
    n_layers = 2
    dropout = 0.2

    # Initialize model
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.00001)

    print("training model...")

    # Train the model
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for texts, labels in train_loader:
            texts, labels = texts.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = model(texts, [len(x) for x in texts])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if rank == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
               

    if rank == 0:
        print("Training completed.")
        torch.save(model.module.state_dict(), 'LSTM/LSTM_model.pth')
        torch.save(my_vocab, 'LSTM/vocab.pth')
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion, rank)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}')

    destroy_process_group()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
