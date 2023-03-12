import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4'])

import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''
    load_data
    Loads data from a sqlite database

    Input:
    database_filepath   file containing the sqlite database
    

    Returns:
    X   input data for training
    Y   labels for the training data
    category_names  list of all the categories which the data is labeled into

    '''
    
    # read in file
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', engine)

    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = list(Y.columns)

    return X, Y, category_names


def tokenize(text):
    '''
    tokenize
    Transforms a string into a list of tokens for NLP training

    Input:
    text    string with input data

    Returns:
    clean_tokens    list of tokenized text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    '''
    build_model
    Builds a sklearn pipeline with data preprocessing and Grid Search configurations for model training

    Returns:
    model_pipeline  sklearn pipeline
    '''

    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])


    # define parameters for GridSearchCV
    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        'clf__estimator__n_estimators': [50, 100]
        #'clf__estimator__min_samples_split': [2, 3, 4]
    }


    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters)


    return model_pipeline


def evaluate_model(model, X_test, Y_test, category_names):

    '''
    evaluate_model
    evaluates a trained model against a set of test data and prints the result
    

    Input:
    model   model to be evaluated
    X_test  input test data
    Y_test  input label data
    category_names  list of categories which the data is classified into
    
    '''

    # output model test results
    y_pred = model.predict(X_test)

    # Clean results, as they can be different from 0 and 1
    y_pred = (y_pred > 0.5) 
    y_test = (Y_test > 0.5) 
    print(classification_report(y_test, y_pred))


def save_model(model, model_filepath):

    '''
    save_model
    Saves a sklearn model in a specific path in the pickle format

    Input:
    model   sklearn model to be saved
    model_filepath  path where the model will be saved
    
    '''

    # Export model as a pickle file
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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