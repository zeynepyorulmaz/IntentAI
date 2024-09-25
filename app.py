from flask import Flask, render_template, request
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('intent_recognition_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the user input from the form
        user_input = request.form['user_input']
        
        # Preprocess and vectorize the input
        vectorized_input = vectorizer.transform([user_input])
        
        # Predict the intent using the trained model
        predicted_intent = model.predict(vectorized_input)[0]
        
        # Render the result on a new page
        return render_template('result.html', intent=predicted_intent, user_input=user_input)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
