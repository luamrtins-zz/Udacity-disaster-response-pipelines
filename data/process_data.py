import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    - Take two csv files 
    - Load them and save as pandas dataframe
    - Returns dataframe merged with both files
    
    Args: 
    - messages_filepath: str  
    - categories_filepath: str

    Returns:
    - merged dataframe: pd.DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on='id')

    return df 


def clean_data(df):
    """
    Clear the dataframe and return it.
    """
    # create a dataframe of individual category columns
    categories = df.categories.str.split(';', expand = True)
    categories.columns = categories.iloc[0]
    categories = categories.rename(columns = lambda x : str(x)[:-2])

    for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].astype(str).str.slice(start=-1)
    categories[column] = pd.to_numeric(categories[column])

    df.drop('categories', axis=1, inplace=True)
    df = df.join(categories)

    # check number of duplicates
    number_duplicates = df.duplicated().sum()

    if number_duplicates != 0:
        df = df.drop_duplicates()
        if df.duplicated().sum() == 0:
            continue
        else:
            print('Not all duplicates were removed.')
    return df


def save_data(df, database_filename):
    """
    Saves dataframe as sql file
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    
    df.to_sql('disaster_response', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()