import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras import Sequential

# Adjust the path to your dataset
dataset_path = 'go_emotions_pre_processed.csv'
data = pd.read_csv(dataset_path)

# Assuming 'data' is your DataFrame and it's already loaded
X = data['text'].values
y = data.drop(['id', 'text'], axis=1).values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_features = 5000  # Number of words in the vocabulary
max_length = 100  # Maximum length of a text sequence

vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=max_length)

# Adapt the text vectorization layer to your text data
vectorize_layer.adapt(X_train)

# Define the model
model = Sequential([
    vectorize_layer,
    Embedding(max_features, 64, input_length=max_length),
    LSTM(64),
    Dense(y.shape[1], activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
print(model.summary())

history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.2)

performance = model.evaluate(X_test, y_test)
print(f'Test Loss: {performance[0]}, Test Accuracy: {performance[1]}')