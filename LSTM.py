import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter
from torchtext.vocab import vocab

# Set device
device = 'cuda:5' if torch.cuda.is_available() else torch.device("mps")

# Model definition
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

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_accuracy, total_precision, total_recall, total_f1 = 0, 0, 0, 0, 0
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts, [len(x) for x in texts])
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            accuracy, (precision, recall, f1), _ = precision_recall_fscore_support(labels.cpu(), predictions.cpu(), average='macro')
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1
    return total_loss / len(loader), total_accuracy / len(loader), total_precision / len(loader), total_recall / len(loader), total_f1 / len(loader)

def main():
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
    encoded_labels = LabelEncoder().fit_transform(y)

    # Create tensors
    text_tensor = [torch.tensor(x) for x in encoded_texts]
    label_tensor = torch.tensor(encoded_labels)

    # Padding sequences to create uniform length
    padded_texts = pad_sequence(text_tensor, batch_first=True)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(padded_texts, label_tensor, test_size=0.2, random_state=42)

    # Create data loaders
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Model parameters
    embedding_dim = 100
    hidden_dim = 256
    output_dim = len(set(y))
    n_layers = 2
    dropout = 0.5

    # Initialize model
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    print("training model...")

    # Train the model
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts, [len(x) for x in texts])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    # Evaluate the model
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}')

    # Save the model and tokenizer
    torch.save(model.state_dict(), 'LSTM/LSTM_model.pth')
    torch.save(my_vocab, 'LSTM/vocab.pth')

    print("Model and vocabulary have been saved.")

if __name__ == '__main__':
    main()
