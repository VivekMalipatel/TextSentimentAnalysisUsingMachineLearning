from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import joblib

# Load the dataset to get an overview and understand its structure
dataset_path = 'go_emotions_pre_processed.csv'
model_filename = 'MultinomialNB_Model.joblib'

df = pd.read_csv(dataset_path)

# Extracting the features and labels
X = df['text'].astype(str)
# Since this is a multi-label classification, we'll consider all emotion columns as labels
y = df.iloc[:, 2:]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(X_train.info())
print(X_test.info())

# Creating a TF-IDF Vectorizer and Multinomial Naive Bayes model pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', OneVsRestClassifier(LinearSVC())),
])

# Fitting the model
pipeline.fit(X_train, y_train)

# Predicting on the test set
y_pred = pipeline.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(accuracy*100,2)}")

# Due to the multi-label nature, let's compute average F1 score across all labels for a more comprehensive evaluation
report= classification_report(y_test, y_pred, target_names=y.columns, zero_division=0)
print(report)

joblib.dump(pipeline, model_filename)