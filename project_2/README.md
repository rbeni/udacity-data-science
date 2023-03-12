# Disaster Response Pipeline Project

## Project Motivation
This project is the capstone of the Data Engineering part of the Data Science Nanodegree. Here I apply the knowledge provided to create pipelines, which integrate the ETL and Model training parts of a project.

The project has the objective to create a pipeline to analyze twitter messages and classify them according to a few categories.an

The ultima goal for the project is an application which allows authorities and organizations to gather information in a faster way about what is needed by a community in an event of a disaster, be it natural or man-made, and also provide the people affected by such disasters a centralized way to communicate what is needed.


## File Description

    app
    | - template
    | |- master.html # main page of web app
    | |- go.html # web app page which displays the results of the analysis
    |- run.py # File that contains the flask code to generate the web app
    data
    |- disaster_categories.csv # data to process
    |- disaster_messages.csv # data to process
    |- process_data.py #Contains the code to extract and the data given in the csv files and save it in a sqlite database
    |- DisasterResponse.db # database containing data cleaned by process_data.py
    models
    |- train_classifier.py #Contains the code that trains a ML model based on the data provided by the ETL step
    |- classifier.pkl # saved model trained by train_classifier.py
    README.md



## Prerequisites
- Python>=3.8.15
- nltk>=3.7
- numpy>=1.23.5
- pandas>=1.5.1
- python-dateutil>=2.8.2
- scikit-learn>=1.1.3
- scipy>=1.9.3
- six>=1.16.0
- SQLAlchemy>=1.4.44
- flask>=2.2.3
- joblib>=1.2.0
- tqdm>=4.64.1
- urllib3>=1.26.12
- wget>=3.2

## Running
To run this project, you will have to:

Note that the commands below are for Python 3.x interpreters. If you are using Python 2.x, use simply python on the beggining

1. Perform the ETL step
`python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. Train the model
`python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the app

Go to the directory containing the app `cd app`, then run the app `run.py`

## License
The code was provided by Udacity and the data used for training by [Appen](https://appen.com/)
