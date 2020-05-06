from flask import Flask, request, jsonify, render_template
from newspaper import Article
app = Flask(__name__)
import numpy as np
import pandas as pd
import pickle
import nltk
nltk.download('punkt')

model = pickle.load(open('model_clickbait_final2.sav', 'rb'))
#model_bias= load_model('Deployment-flask-master/model_bias_final2.h5')

def get_article(url):
    title = []
    text = []
    summary = []
    data = {"title": title, "text": text, "summary": summary}
    article = Article(url, language="en")  # en for English
    article.download()
    article.parse()
    article.nlp()
    title.append(article.title)
    text.append(article.text)
    summary.append(article.summary)
    data = {"title": title, "text": text, "summary": summary}
    return pd.DataFrame(data)

def predict_clickbait(pipeline, tdf):
  out={0:"Non-Clickbait", 1:"Clickbait"}
  pred=out[pipeline.predict(tdf.text)[0]]
  return pred

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = int_features[0]
    data=get_article(final_features)
    prediction = predict_clickbait(model, data)

    return render_template('index.html', prediction_text= str(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
