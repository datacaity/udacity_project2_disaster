import sys
import pandas as pd
import sqlite3
from pandasql import sqldf
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    reads in two datasets, merges, removes duplicates
    """
    # 'disaster_messages.csv'
    messages = pd.read_csv(messages_filepath)
    messages = messages.drop_duplicates(keep='first')
    # 'disaster_categories.csv'
    categories = pd.read_csv(categories_filepath)
    categories = categories.drop_duplicates(keep='first')
    df = pd.merge(messages, categories, on='id', how='left')
    return df



def clean_data(df):
    """
    splits out categories into usable columns with binary values
    """
    categories_new = df['categories'].str.split(';', expand=True)
    row = categories_new.iloc[0]
    category_colnames = row.str.split('-').str[0]
    categories_new.columns = category_colnames
    for column in categories_new:
        # set each value to be the last character of the string
        categories_new[column] = categories_new[column].str[-1]
        # convert column from string to numeric
        categories_new[column] = pd.to_numeric(categories_new[column], errors='coerce')
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories_new], axis=1, sort=False)
    return df


def save_data(df, database_filename):
    """
    saves cleaned data into a table in named database
    """
    # 'sqlite:///DisasterResponseProject.db'
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Msg_Cat', engine, index=False)
      


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