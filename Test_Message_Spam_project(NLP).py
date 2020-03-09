import sys
import pandas as pd
import numpy
import nltk
import sklearn
dataset = pd.read_table("C:\\Users\\Mohsin\Desktop\\SMSSpamCollection",header=None,encoding='utf-8')
#Checking class distributions i.e how many spam and how many ham
classes = dataset[0]
print(classes.value_counts())
#Encoding the ham/spam column
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder
encoder = LabelEncoder()
Y = encoder.fit_transform(classes)
text_messages = dataset[1]
#Cleaning the data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
# use regular expressions to replace email addresses, URLs, phone numbers, other numbers
# Replace email addresses with 'email'
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddress') 
# Replace URLs with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')
# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
processed = processed.str.replace(r'£|\$', 'moneysymb')
# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumbr')
# Replace numbers with 'numbr'
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
# Remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')
# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')
# Remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')
#Changing the words to lowercase
processed = processed.str.lower()
#Removing stopped words
stop_words = set(stopwords.words('english'))
processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
# Remove word stems using a Porter stemmer
processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))
from nltk.tokenize import word_tokenize
all_words = []
for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(processed).toarray()   

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
#Fitting Classifier to the data set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train,Y_train)
#Presiction of Navie Bayes
Y_pred = classifier.predict(X_test)
#Creating hte confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)


 