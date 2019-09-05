from flask import Flask,render_template,url_for,request
import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import string 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



app=Flask(__name__)

@app.route('/')
def home():
		return render_template('home.html')
	


@app.route('/predict',methods=['GET', 'POST'])
def predict():
	X_test=pd.read_csv('data/Reviews.csv')  #This is just for our input to append and convert it into countvectors
	X_train=pd.read_csv('data/X_train.csv')  #After processing my data
	Y_train = pd.read_csv('data/Y_train.csv') #same target variable
	if request.method=='POST':
		#Taking inputs 
		input1= request.form.get("input1")
		
		#s=["Los Angeles to Santiago de Chile. Overall good flight. Relatively comfortable seats, polite and smiling staff, ok choice of movies.Aside these two points, good flight experience."]
		s=[input1]
		#appending our review to X_test dataframe
		dx=pd.DataFrame({"Reviews":s})
		X_test = X_test.append(dx)
		count_vect = CountVectorizer()
		#converting into numbers
		X_train_vector = count_vect.fit_transform(X_train['Reviews'])

		X_test_vector = count_vect.transform(X_test['Reviews'])

		algo = MultinomialNB().fit(X_train_vector,Y_train)
		prediction = algo.predict(X_test_vector)
		#getting last row that is our review
		prediction = prediction[len(prediction)-1]
		print(prediction)

		if prediction == 0 :
			myprediction = 'Negative Review'
		else :
			myprediction =' Positive Review'
		return render_template('result.html',my_classifier=myprediction)


if __name__ == '__main__':
	app.run(debug=True)