import spacy
import re
from flask import Flask, request
from flask_ask import Ask, statement
from symspellpy import SymSpell, Verbosity


# Initialize Flask and Flask-Ask
app = Flask(__name__)
ask = Ask(app, '/')

def correct_sentence(patient_sentence: str) -> str:
    # Inicializar SymSpell
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

    # Ruta al archivo descargado
    dictionary_path = "assets/frequency_dictionary_en_82_765.txt"
    bigram_path = "assets/frequency_bigramdictionary_en_243_342.txt"

    # Cargar el archivo de diccionario
    if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
        raise ValueError("No se pudo cargar el diccionario")
    if not sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2):
        raise ValueError("No se pudo cargar el bigram")

    # Usar lookup_compound para corregir la frase completa
    suggestions = sym_spell.lookup_compound(patient_sentence, max_edit_distance=2)

    # Devolver la mejor sugerencia
    if suggestions:
        return suggestions[0].term
    else:
        return patient_sentence  # Si no hay sugerencias, devolver original

def calculate_semantic_similarity(patient_text,corrected_text) -> float:
    # Carga el modelo de spaCy
    nlp = spacy.load("en_core_web_md")

    # Representar frases como objetos spaCy
    doc1 = nlp(corrected_text)
    doc2 = nlp(patient_text)

    # Calcular similitud semántica
    similarity = doc1.similarity(doc2)
    
    return similarity

@app.route("/", methods=["GET"])
def test_clasification():
    # Prueba del sistema
    test_text = "Can yu read this mesage despite the horible speling mistakes"
    
    correct_text = correct_sentence(test_text)
    
    print(f"Texto corregido: {correct_text}")

    similarity = calculate_semantic_similarity(test_text, correct_text)
    print(f"Similitud semántica: {similarity:.2f}")

    return "Hello WORLD"

@ask.intent('ClassificationIntent', mapping={'user_text': 'UserText'})
def classify_intent(user_text):
    correct_text = correct_sentence(user_text)
    
    print(f"Texto corregido: {correct_text}")

    similarity = calculate_semantic_similarity(user_text, correct_text)
    print(f"Similitud semántica: {similarity:.2f}")

    if similarity > 0.99:
        speech_text = f"Your text was perfectly understood. No corrections were necessary. Your aphasia level is classified as none."
    elif 0.90 <= similarity <= 0.99:
        speech_text = f"Your text has been slightly corrected. The corrected text is: '{correct_text}'. Your aphasia level is classified as mild."
    elif 0.70 <= similarity < 0.90:
        speech_text = f"Your text required moderate corrections. The corrected text is: '{correct_text}'. This indicates a moderate level of aphasia."
    else:
        speech_text = f"Your text required significant corrections. The corrected text is: '{correct_text}'. This suggests a severe level of aphasia."
    
    return statement(speech_text).simple_card('Classification', speech_text)

if __name__ == '__main__':
    app.run()
