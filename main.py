from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import os
import math
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
Bootstrap(app)

# SENTIMENT ANALYSER FUNCTION (using VADER)


def sentiment_analyster(text):
    result = sid.polarity_scores(text)['compound']

    if result > 0.5:
        return f"RESULT: POSITIVE TEXT  ||  CONFIDENCE: {math.floor(result*100)}%"
    else:
        return f"RESULT: NEGATIVE TEXT  CONFIDENCE: {math.floor(result*-100)}%"


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        prediction = sentiment_analyster(text)
        print(prediction)
        return render_template('index.html', prediction=prediction, prediction_complete=True)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
