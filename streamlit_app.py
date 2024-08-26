import streamlit as st
import joblib
import string
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the trained model pipeline (which includes both vectorizer and model)
with open("nb12.pkl", "rb") as model_file:
    pipeline = joblib.load(model_file)

# Load label encoder if needed (only if mapping is required separately)
with open("final_tickets_encoder.pkl", "rb") as encoder_file:
    label_encoder = joblib.load(encoder_file)

# Define the mapping for the dependent variable (if necessary)
def map_prediction(prediction):
    # Uncomment and use this function if label_encoder is used separately.
    return label_encoder.inverse_transform(prediction)[0]
    return prediction[0]  # Assuming pipeline includes label mapping

# Text cleaning function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove newlines
    text = text.replace("\n", " ")
    
    # Remove multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    custom_words = {'productpurchased', 'please assist', 'im'}  # Add custom words/phrases here
    all_stop_words = stop_words.union(custom_words)
    
    # Remove stop words and custom words
    text = ' '.join([word for word in text.split() if word not in all_stop_words])
    
    return text

# Function to classify the ticket
def classify_ticket(ticket_description):
    # Clean the text before prediction
    cleaned_description = clean_text(ticket_description)
    
    # Predict using the pipeline
    prediction = pipeline.predict([cleaned_description])
    return map_prediction(prediction)

def main():
    # Set the page title and layout
    st.set_page_config(page_title="Support Ticket Classification", layout="centered")

    # Custom CSS for modern styling
    st.markdown(
        """
        <style>
        .header {
            background-color: #376da3;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .header h2 {
            color: #ecf0f1;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .description {
            text-align: center;
            font-size: 1.2rem;
            margin-top: 20px;
            color: #2c3e50;
        }
        .stTextArea textarea {
            border: 2px solid #3498db;
            border-radius: 10px;
            padding: 12px;
            font-size: 1.1rem;
            color: #2c3e50;
        }
        .stButton button {
            background-color:  #376da3;
            color: white;
            padding: 12px 24px;
            font-size: 1.2rem;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #2980b9;
        }
        </style>
        <div class="header">
            <h2>Support Ticket Classification</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Brief description and instructions
    st.write(
        """
        <p class="description">Welcome! 🎫</p>
        <p class="description">Enter the ticket description below, and click "Predict" to see its classification.</p>
        """,
        unsafe_allow_html=True
    )

    # Text input for ticket description
    ticket_description = st.text_area(
        "Enter Ticket Description", 
        help="Type the ticket description you want to classify."
    )

    # Predict button
    if st.button("Predict", help="Click to classify the ticket description"):
        if ticket_description.strip():
            result = classify_ticket(ticket_description)
            st.success(f'The classification is: **{result}**')
        else:
            st.error("Please enter a ticket description.")


if __name__ == '__main__':
    main()
