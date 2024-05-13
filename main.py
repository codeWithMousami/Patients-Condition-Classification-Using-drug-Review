import streamlit as st
from bs4 import BeautifulSoup
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Import image library
from PIL import Image
# Load image with relative path
image = Image.open('image.jpg')



# Load the pre-trained models and data
tfidf_vectorizer = joblib.load('tfidfvectorizer.pkl')
passive_model = joblib.load('passmodel.pkl')
df = pd.read_csv('drugsComTest_raw.csv')  # Load your dataset here

# Define the top drugs extractor function
def top_drugs_extractor(condition):
    df_top = df[(df['rating']>=9) & (df['usefulCount']>=100)].sort_values(by=['rating', 'usefulCount'], ascending=[False, False])
    drug_lst = df_top[df_top['condition'] == condition]['drugName'].head(3).tolist()
    return drug_lst

# Define the review_to_words function for cleaning input text
def review_to_words(raw_review):
    lemmatizer = WordNetLemmatizer()
    stop = stopwords.words('english')
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not w in stop]
    lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    return ' '.join(lemmatized_words)

# Define the Streamlit app
def main():
    st.title('Patient Condition Classification')
    st.image(image, use_column_width=True)

    # Input box for patient symptoms
    symptoms = st.text_area('Enter patient symptoms:')
    cleaned_symptoms = review_to_words(symptoms)

    # Display preprocessed data
    st.subheader('Preprocessed Data:')
    st.write(cleaned_symptoms)

    if st.button('Classify Condition'):
        tfidf_symptoms = tfidf_vectorizer.transform([cleaned_symptoms])
        condition = passive_model.predict(tfidf_symptoms)[0]

        st.subheader(f'Predicted Condition: {condition}')

        # Output top 3 suggested drugs
        top_drugs = top_drugs_extractor(condition)
        st.subheader('Top 3 Suggested Drugs:')
        for drug in top_drugs[:3]:
            st.write(drug)

# Run the app
if __name__ == "__main__":
    main()