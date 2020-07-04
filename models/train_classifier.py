# download necessary NLTK data
import nltk
nltk.download(['punkt','wordnet', 'stopwords','averaged_perceptron_tagger'])

# import libraries
import sys
import pandas as pd
import re

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import pickle




def load_data(database_filepath):
    """
    Read a sql file and returns the vars X(pd.Series), y(pd.DataFrame), category_names(list)
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', con = engine)
    X = df['message']
    y = df.drop(["id", "message","original", "genre"], axis = 1)
    category_names = list(y.columns)
    return X, y, category_names


def tokenize(text):
    """
    Function that takes messages and returns a list of cleaned words in the message
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = [ word for word in word_tokenize(text) if word not in stop_words]
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """
    Takes in the message column and returns classification results on 36 categories 
    in the dataset
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
   
    parameters = {
        'vect__max_df': [0.5, 0.75],
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [10, 20],
        'clf__estimator__min_samples_split': [2, 5],
        'clf__estimator__max_features': ['auto', 'sqrt'],
        'clf__estimator__max_depth': [10, 25, 50, 75, 100],
    }
  
                
#     {'vect__min_df': [1, 5],
#                   'tfidf__use_idf':[True, False],
#                   'clf__estimator__n_estimators':[10, 25], 
#                   'clf__estimator__min_samples_split':[2, 5, 10]}
    
    # Create grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 10)
    

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Predict on the X_test and returns test accuracy, recall, precision and F1 Score
    """

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names = category_names))


def save_model(model, model_filepath):
    """"
    Saves the trained model into a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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