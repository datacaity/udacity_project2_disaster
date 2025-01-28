# Disaster Response Pipeline Project
### Table of Contents

1. [Project Overview](#projectoverview)
2. [File Descriptions](#files)
3. [Instructions](#instructions)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Overview <a name="projectoverview"></a>

This project was done as part of Udacity's Data Science Nanodegree. The purpose of this project 
was to take in data from Appen regarding disaster messages and their corresponding disaster categories
in order to build a model that can process new disaster messages and report them appropriately.

This results in a web app interface where a message can be typed into the search and then result in the
appropriate disaster categories. There are screenshots of an example in the main folder.

## File Descriptions <a name="files"></a>
 
 Folder: app
	run.py - python script that launches web app. See instructions below.
	templates folder : go.html and master.html - used to launch web app
	
Folder: data
	disaster_categories.csv - Appen provided dataset with categories for id'd disaster messages
	disaster_messages.csv - Appen provided dataset with disaster messages and their accompanying id.
	DisasterResponseProject.db - database to hold Msg_Cat table from processed python files
	ETL Pipeline Preparation.ipynb - jupyter notebook working through data processing and merging of data files above.
	process_data.py - python script to run in terminal to process data for web app. See instructions below.
	
Folder: models
	classifier.pkl - model
	DisasterResponseProject.db - database to hold Msg_Cat table from processed python files
	ML Pipeline Preparation.ipynb - jupyter notebook working through natural language processing and model building.
	train_classifier.py - python script to run in terminal to model data for web app. See instructions below.

### Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponseProject.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponseProject.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
	`cd app`
    `python run.py`

3. Go to http://127.0.0.1:3001

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

This data was provided courtesy of Appen and Udacity. 
[here](https://www.appen.com/).