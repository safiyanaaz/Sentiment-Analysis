'''This is the processing done for Training the Model in text'''
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
import pickle


contractions = { 
"h." :"hours", "hr." : "hours" , "hr" : "hours","ain't": "am not / are not / is not / has not / have not","aren't": "are not / am not",
"can't": "cannot","can't've": "cannot have","'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have",
"didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not",
"haven't": "have not","he'd": "he had / he would","he'd've": "he would have","he'll": "he shall / he will","he'll've": "he shall have / he will have","he's": "he has / he is",
"how'd": "how did","how'd'y": "how do you","how'll": "how will","how's": "how has / how is / how does","I'd": "I had / I would",
"I'd've": "I would have","I'll": "I shall / I will","I'll've": "I shall have / I will have","I'm": "I am","I've": "I have","isn't": "is not",
"it'd": "it had / it would","it'd've": "it would have","it'll": "it shall / it will","it'll've": "it shall have / it will have","it's": "it has / it is",
"let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not","mightn't've": "might not have","must've": "must have","mustn't": "must not",
"mustn't've": "must not have","needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
"oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not","shan't've": "shall not have","she'd": "she had / she would","she'd've": "she would have",
"she'll": "she shall / she will","she'll've": "she shall have / she will have","she's": "she has / she is","should've": "should have",
"shouldn't": "should not","shouldn't've": "should not have","so've": "so have","so's": "so as / so is","that'd": "that would / that had",
"that'd've": "that would have","that's": "that has / that is","there'd": "there had / there would","there'd've": "there would have","there's": "there has / there is","they'd": "they had / they would","they'd've": "they would have",
"they'll": "they shall / they will","they'll've": "they shall have / they will have","they're": "they are",
"they've": "they have","to've": "to have","wasn't": "was not","we'd": "we had / we would","we'd've": "we would have","we'll": "we will",
"we'll've": "we will have","we're": "we are","we've": "we have","weren't": "were not",
"what'll": "what shall / what will","what'll've": "what shall have / what will have","what're": "what are","what's": "what has / what is","what've": "what have","when's": "when has / when is","when've": "when have",
"where'd": "where did","where's": "where has / where is","where've": "where have","who'll": "who shall / who will","who'll've": "who shall have / who will have",
"who's": "who has / who is","who've": "who have","why's": "why has / why is","why've": "why have","will've": "will have","won't": "will not","won't've": "will not have","would've": "would have",
"wouldn't": "would not","wouldn't've": "would not have","y'all": "you all","y'all'd": "you all would","y'all'd've": "you all would have",
"y'all're": "you all are","y'all've": "you all have","you'd": "you had / you would","you'd've": "you would have","you'll": "you shall / you will","you'll've": "you shall have / you will have",
"you're": "you are","you've": "you have"}

X=pd.read_csv('data/Reviews.csv')
y=pd.read_csv('data/recommended.csv')
reviews = X['Reviews'].tolist()
#print(reviews)
#Replacing contractions
reviews_after_contractions = []
for item in reviews:
    new_item = ' '.join(str(contractions.get(word,word)) for word in item.split())
    reviews_after_contractions.append(new_item)
#print(reviews_after_contractions)
	#Removing unwanted punctuation
docs_no_punctuation = []
regex = r"[\x96\x92\x94x93\x85\t\x80\x93]"

for item in reviews_after_contractions:
    text = item.translate(str.maketrans('','',string.punctuation))
    text = re.sub(regex,"",text)
    docs_no_punctuation.append(text)
#print(docs_no_punctuation)
#Tokenize 

from nltk.tokenize import word_tokenize

tokens =[]
for  item in docs_no_punctuation:
    tokens.append(word_tokenize(item))
#print(tokens)
#Remove stop words 

tokens_no_stopwords=[]

for doc in tokens:
    new_vector = []
    for word in doc:
        if not word in stopwords.words('english'):
            new_vector.append(word)
    tokens_no_stopwords.append(new_vector)

#print(tokens_no_stopwords)
tokens_no_stopwords= [' '.join(x) for x in tokens_no_stopwords]
X['Reviews'] = tokens_no_stopwords
print(X.shape)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.3 , random_state = 0)

'''After processing Save this X to csv to use it in flask app'''
X_train.to_csv('X_train.csv')
Y_train.to_csv('Y_train.csv')