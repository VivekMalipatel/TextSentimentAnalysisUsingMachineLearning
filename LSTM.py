import pandas as pd
from keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load your pre-processed dataset
data_path = 'pre_processed_text.csv'
data = pd.read_csv(data_path)

# Prepare the text and labels
texts = data['text'].values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['label'])
labels = to_categorical(labels)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Text Vectorization
max_features = 10000
sequence_length = 1000

vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

vectorize_layer.adapt(X_train)

# Build the model
model = Sequential([
    vectorize_layer,
    Embedding(max_features + 1, 64, mask_zero=True),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

# Save the model
model.save('LSTM')  # SavedModel format

# Optional: save as HDF5
model.save('LSTM/my_model.h5')  # HDF5 file

# Function to plot graphs
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

# Plot accuracy and loss
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.show()
