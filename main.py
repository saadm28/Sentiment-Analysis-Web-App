# Import libraries
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from transformers import pipeline
from flask_sqlalchemy import SQLAlchemy
import os
import nltk
from nltk.tokenize import sent_tokenize
import json


nltk.download('punkt')  # Download the necessary data for sentence tokenization

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
Bootstrap(app)


# connect to database
DB_NAME = 'sentiment.db'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
db = SQLAlchemy(app)

# create database model
class Sentiment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), nullable=False)
    label = db.Column(db.String(10), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    sentiments = db.Column(db.String(500), nullable=False)

# create database if it doesn't exist
if not os.path.exists(DB_NAME):
    with app.app_context():
        db.create_all()
    print('Database created successfully!')


# load the sentiment analysis model
sentiment_pipeline = pipeline('sentiment-analysis')

# this function should determine and return the sentiment of the text 
def analyze_sentiment(text):
    # Split text into sentences using nltk's sentence tokenizer
    sentences = sent_tokenize(text)

    # create an empty dictionary to store the sentiment of each sentence
    # where the key is the sentence and the value is the sentiment
    sentiments = []

    # determine the sentiment of each sentence
    for sentence in sentences:
        sentiment_result = sentiment_pipeline(sentence)[0]
        sentiments.append({
            'sentence': sentence,
            'label': sentiment_result['label'],
            'confidence': round(sentiment_result['score'] * 100, 2)
        })


    # get overall sentiment score
    label = sentiment_pipeline(text)[0]['label']
    score = round(sentiment_pipeline(text)[0]['score'] * 100, 2)
    
    # save the text, label and score to the database
    # serialize the sentiments dictionary to JSON
    json_sentiments = json.dumps(sentiments)
    new_sentiment = Sentiment(text=text, label=label, score=score, sentiments=json_sentiments)
    db.session.add(new_sentiment)
    db.session.commit()

    # return the overall sentiment
    return {'label': label, 'score': score, 'sentiments': sentiments}
   

# home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        # Call analyze_sentiment to get the sentiment of the text
        analysis_result = analyze_sentiment(text)
        prediction = analysis_result['label']
        confidence = analysis_result['score']
        sentiments = analysis_result['sentiments']
        return render_template('index.html', prediction=prediction, confidence=confidence, sentiments=sentiments, prediction_complete=True)
    return render_template('index.html')


# run the app
if __name__ == '__main__':
    app.run(debug=True)


