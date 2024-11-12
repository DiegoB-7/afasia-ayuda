from flask import Flask, request
from flask_ask import Ask, statement
from transformers import pipeline

# Initialize Flask and Flask-Ask
app = Flask(__name__)
ask = Ask(app, '/')

# Load Hugging Face pre-trained text classification pipeline
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Predict function for Flask
def predict(text):
    # Get classification prediction
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    
    # Define aphasia level and recommendation based on label and score
    if label == "NEGATIVE" and score > 0.7:
        level = "alto"
        recommendation = "Se recomienda una evaluación clínica detallada para posibles problemas de lenguaje."
    elif label == "NEGATIVE":
        level = "moderado"
        recommendation = "Se sugiere revisar la estructura del lenguaje y buscar asesoría si es necesario."
    else:
        level = "bajo"
        recommendation = "La estructura del lenguaje parece adecuada."
    
    return level, recommendation

@app.route("/", methods=["GET"])
def test_clasification():
    
    return f"Hola mundo"

@ask.intent('ClassificationIntent', mapping={'user_text': 'UserText'})
def classify_intent(user_text):
    level, recommendation = predict(user_text)
    speech_text = f"Nivel de afasia detectado: {level}. {recommendation}"
    return statement(speech_text).simple_card('Classification', speech_text)

if __name__ == '__main__':
    app.run()
