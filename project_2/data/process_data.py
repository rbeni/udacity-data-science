import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    #Load files
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    
    #Merge datasets
    df = pd.concat([messages_df, categories_df], axis=1)
    return df
    

def clean_data(df):
    
    #Split categories into separate category columns
    categories_names = df['categories'].str.split(';', expand=False).iloc[0]
    for i, cat_name in enumerate(categories_names):
        new_name = cat_name.split('-')[0]
        categories_names[i] = new_name

    #Create a dataframe 
    categories_df = df['categories'].apply(lambda x: pd.Series(x.split(';')))

    #Apply new names to the dataframe
    categories_df.columns = categories_names

    #Clean values of dataframe, from string to integer
    for column in categories_df:
        categories_df[column] = categories_df[column].apply(lambda x: x.split('-')[-1]).astype(int)

    #Replace cleaned categories dataframe on main dataset
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories_df], axis=1)

    #Remove duplicates
    df = df[df.duplicated(keep='first')]

    return df


def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False)


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