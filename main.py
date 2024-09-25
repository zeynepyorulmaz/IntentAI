import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 1: Data Selection
# Load and parse the JSON dataset
with open('Intent.json', 'r') as f:
    data = json.load(f)

# Prepare the dataset
texts = []
labels = []

for intent in data['intents']:
    for text in intent['text']:
        texts.append(text)
        labels.append(intent['intent'])

# Convert the data into a DataFrame
df = pd.DataFrame({'text': texts, 'intent': labels})

# Step 2: Text Preprocessing
# Text Preprocessing using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')

# Step 3: Feature Extraction
X = vectorizer.fit_transform(df['text'])
y = df['intent']

# Step 5: Training the Model
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training using Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Step 7: Model Deployment
# Save the model and vectorizer for deployment
joblib.dump(model, 'intent_recognition_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Deploying the model (for real-time use)
# Example: Load the model and use it for prediction

# Load the saved model and vectorizer
loaded_model = joblib.load('intent_recognition_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example prediction
sample_text = ["Tell me a gossip"]
sample_vectorized = loaded_vectorizer.transform(sample_text)
predicted_intent = loaded_model.predict(sample_vectorized)

print(f"Predicted Intent for '{sample_text[0]}': {predicted_intent[0]}")
