import streamlit as st
import joblib

model = joblib.load('imdb_sentiment_model.pkl')
vectorizer = joblib.load('imdb_vectorizer.pkl')

st.title('IMDb Movie Review Sentiment Analysis')

review = st.text_area('Please Enter Movie Review:')

if st.button('Predict'):
    if review:
        review = review.lower()
        review_vectorized = vectorizer.transform([review])
        
        prediction = model.predict(review_vectorized)[0]
        
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        st.write(f'Sentiment Movie Review: **{sentiment}**')
    else:
        st.write("Please Enter Text")
