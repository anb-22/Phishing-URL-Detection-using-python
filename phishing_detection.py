import numpy as np
import feature_extraction
from sklearn.ensemble import RandomForestClassifier as rfc
#from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression as lr
from flask import jsonify 


def getResult(url):

    #Importing dataset 
    filen = 'dataset.csv'
    r = open(filen,'rt')
    data = np.loadtxt(r, delimiter = ",") # loading the dataset

    #lets seperating features and labels
    X = data[: , :-1]
    y = data[: , -1]

    #Seperating training features, testing features, training labels & testing labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    clf = rfc()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score*100)  # accuracy score

    X_new = []

    X_input = url # checking for catogery
    X_new=feature_extraction.generate_data_set(X_input) # extracting features of given url
    X_new = np.array(X_new).reshape(1,-1)  # converting 

    try:
        prediction = clf.predict(X_new)
        if prediction == -1:
            return "Omg!!!.. its Phishing Url"
        else:
            return "hureeh!!....its a Genuine Url"
    except:
        return "Omg!!... its a Phishing Url"
