from flask import Flask, request, render_template, jsonify
import joblib
import keras.models
import pickle
import numpy as np
import os
import re
import sys
import json

sys.path.append(os.path.abspath('/IRIS-FLASK(works)'))




#NB = pickle.load(open('iris_flask.pickle','rb'))
app = Flask(__name__)

#estimator = joblib.load('iris_flask.pickle')
#target_names = ['setosa','versicolor','virginica']
#global model, graph
#model, graph = init()

@app.route('/')
def index():
	return render_template('index2.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,4)
    loaded_model = pickle.load(open("iris_flask.pickle","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result', methods = ['POST', 'GET'])
def predict():
	if request.method == 'POST':
		sl = request.form['sl']
		sw = request.form['sw']
		pl = request.form['pl']
		pw = request.form['pw']

		sample_data = [sl,sw,pl,pw]
		data = np.array(sample_data).reshape(1,-1)
		loaded_model = joblib.load(open("iris_svm.pickle","rb"))
		result = loaded_model.predict(data)
		predictions = result[0]		

		if int(predictions) == 0:
			predictions = 'setosa'
		elif int(predictions) == 1:
			predictions = 'versicolor'
		else:
			predictions = 'virginica'

	return render_template('result.html',sl = sl,sw = sw, pl = pl, pw = pw,predictions = predictions)






	#if request.method == 'POST':
#	data = request.get_json()
		#predict_request = request.form['sepallength','sepalwidth','petallength','petalwidth']
#	predict_request = [data['sl'],data['sw'],data['pl'],data['pw']]
#	predict_request = np.array([predict_request])
#	predict_request = predict_request.reshape(1,-1)

#	with graph.as_default():
#	predictions = NB.predict(predict_request)
#		  response = jsonify({'setosa': str(y_pred[0][0]),'versicolor': str(y_pred[0][1]),'virginica': str(y_pred[0][2])})
#		return response 


#		return render_template('result.html', predictions = predictions)


if __name__ == '__main__':
	app.run(host = "0.0.0.0",port= 8078)