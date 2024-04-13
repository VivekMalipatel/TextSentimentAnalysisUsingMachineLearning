import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

def main():
    # Load the pre-processed dataset
    data_path = 'pre_processed_text.csv'
    data = pd.read_csv(data_path)

    # Extract features and labels
    X = data['text']
    y = data['label']

    # Vectorize the text data with unigrams, bigrams, and trigrams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30000)
    X_vectorized = vectorizer.fit_transform(X)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # Train the Naive Bayes model with adjusted hyperparameter alpha
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)

    # Predict on the test set and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

    # Save the trained model and vectorizer
    joblib.dump(model, 'Baseline_Models/Naive_Bayes/NaiveBayes_model_files/naive_bayes_model.joblib')
    joblib.dump(vectorizer, 'Baseline_Models/Naive_Bayes/NaiveBayes_model_filestfidf_vectorizer.joblib')

    print("Model and vectorizer have been saved.")

if __name__ == '__main__':
    main()
