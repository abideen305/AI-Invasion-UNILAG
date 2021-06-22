import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('medical_insurance.pkl', 'rb'))



@app.route('/')
@app.route('/home')
def hello_world():
    return render_template('index.html', title= 'home')
@app.route('/about')
def about():
	return render_template('about.html', title='About')
@app.route('/stories')
def stories():
	return render_template('stories.html', title='About')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    return render_template('index.html', prediction_text='The customer status is {}'.format(prediction))
 

if __name__ == '__main__':
    app.run(debug = True)