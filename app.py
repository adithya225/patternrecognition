import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('randomforest.pkl', 'rb'))
lb = pickle.load(open('lb', 'rb'))
lb1 = pickle.load(open('lb1', 'rb'))
lb2 = pickle.load(open('lb2', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [str(x) for x in request.form.values()]
    print(features)
    engine_power = float(features[1])
    door_count = int(features[3])
    seat_count = int(features[4])
    final_features = []
    final_features.append(lb.transform([features[0]])[0])
    final_features.append(engine_power)
    final_features.append(lb1.transform([features[2]])[0])
    final_features.append(door_count)
    final_features.append(seat_count)
    final_features.append(lb2.transform([features[5]])[0])
    print(len(final_features))
    prediction = model.predict([final_features])
    output = round(prediction[0], 2)    
    return render_template('index.html',prediction_text = output)
 


if __name__ == "__main__":
    app.run(debug=True)
