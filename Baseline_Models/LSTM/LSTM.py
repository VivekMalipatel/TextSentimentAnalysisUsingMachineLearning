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
import torch.nn.functional as F

# Set device
device = 'cuda:5' if torch.cuda.is_available() else torch.device("mps")

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output, lengths):
        # lstm_output is of shape [batch_size, seq_len, hidden_dim]
        scores = self.linear(lstm_output).squeeze(2)  # [batch_size, seq_len]
        max_len = max(lengths)
        for i, length in enumerate(lengths):
            if length < max_len:
                scores[i, length:] = -1e9  # Large negative value for softmax
        weights = F.softmax(scores, dim=1)
        # Apply weights
        weighted = torch.sum(lstm_output * weights.unsqueeze(2), dim=1)
        return weighted, weights

# Model definition
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # Unpack output
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # Apply attention
        attn_output, attn_weights = self.attention(output, text_lengths)
        out = self.fc(attn_output)
        return out

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_accuracy, total_precision, total_recall, total_f1 = 0, 0, 0, 0, 0
    num_batches = len(loader)
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
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

def main():
    # Load the pre-processed dataset
    data_path = 'Dataset/Training_dataset/pre_processed_text.csv'
    data = pd.read_csv(data_path)
    
    # Extract features and labels
    X = data['text'].astype(str)
    y = data['label']

    # Create a vocabulary
    token_counts = Counter(word for sentence in X for word in sentence.split())
    my_vocab = vocab(token_counts, specials=['<pad>'], min_freq=2)
    my_vocab.set_default_index(my_vocab['<pad>'])
    vocab_size = len(my_vocab)

    # Describe vocab_size, token_counts, and sample of my_vocab
    print("vocab_size:", vocab_size)
    print("Number of unique tokens:", len(token_counts))
    print("Sample of token_counts:", token_counts.most_common(10))

    # Encode text and labels
    encoded_texts = [[my_vocab[word] for word in text.split() if word in my_vocab] for text in X]
    encoded_labels = y

    # Create tensors
    text_tensor = [torch.tensor(x, dtype=torch.long) for x in encoded_texts]
    label_tensor = torch.tensor(encoded_labels)

    # Describe text_tensor
    print("text_tensor shape:", len(text_tensor))
    print("text_tensor dtype:", text_tensor[0].dtype)
    print("text_tensor sample:", text_tensor[0])

    # Padding sequences to create uniform length
    padded_texts = pad_sequence(text_tensor, batch_first=True, padding_value=my_vocab['<pad>'])

    # Describe padded_texts
    print("padded_texts shape:", padded_texts.size())
    print("padded_texts dtype:", padded_texts.dtype)
    print("padded_texts sample:", padded_texts[0])

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(padded_texts, label_tensor, test_size=0.2, random_state=42)

    # Describe X_train and X_test
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("X_train sample:", X_train[0])
    print("X_test sample:", X_test[0])

    # Create data loaders
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    # Model parameters
    embedding_dim = 100
    hidden_dim = 256
    output_dim = len(set(y))
    n_layers = 2
    dropout = 0.2

    # Initialize model
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout).to(device)
    model.embedding.weight.data[my_vocab['<pad>']] = torch.zeros(embedding_dim)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.00001)

    print("training model...")

    # Train the model
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts, [len(x) for x in texts])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs}, Average Epoch Loss: {average_loss:.4f}')
     
    # Save the model and tokenizer
    torch.save(model.state_dict(), 'Baseline_Models/LSTM/LSTM_model_files/LSTM_model.pth')
    torch.save(my_vocab, 'Baseline_Models/LSTM/LSTM_model_files/vocab.pth')

    # Evaluate the model
    print("\n Final model evaluation metrics :")
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}')

    print("Model and vocabulary have been saved.")

if __name__ == '__main__':
    main()
