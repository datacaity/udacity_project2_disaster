import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import re
import pickle
import nltk
nltk.download(['punkt', 'punkt_tab', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    load the filepath for the database
    return the data needed for modeling
    
    """
    dbname = 'sqlite:///' + database_filepath
    # 'sqlite:///../data/DisasterResponseProject.db'
    engine = create_engine(dbname)
    df = pd.read_sql('SELECT * FROM Msg_Cat',engine)
    X = df['message']
    y = df.iloc[:, 4:]
    cat_list = list(df.iloc[:, 4:])
    return X, y, cat_list


def tokenize(text):
    """
    tokenize and clean messages text
    return cleaned text
    
    """
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    building pipeline 
    """
    pipeline = Pipeline([ 
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)), 
        ('tfidf', TfidfTransformer()), 
        ('clf', MultiOutputClassifier(estimator = RandomForestClassifier(n_jobs = -1))) 
        ]) 
    parameters = {
    'clf__estimator__max_depth': [10], 
    'clf__estimator__min_samples_leaf': [2, 5, 10],
    'clf__estimator__n_estimators': [10, 20]
    }

    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, y_test, cat_list):
    """
    print out model results
    """
    y_pred = model.predict(X_test)
    for category in range(len(cat_list)-1):
        print(cat_list[category], 'classification')
        print(classification_report(y_test.iloc[:, category].values, y_pred[:, category]))


def save_model(model, model_filepath):
    """
    save model as pickle file
    """
    # 'classifier.pkl'
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, cat_list = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, cat_list)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()