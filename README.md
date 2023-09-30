This this a project to complete the competition published on Kaggle: 
Sentiment Analysis on Movie Reviews - Classify the sentiment of sentences from the Rotten Tomatoes dataset

The aim of this competition is to categorize 5 different sentiment about a movie from Rotten Tomatoes website. The original dataset is originally collected from Rotten Tomatoes by Pang and Lee.

The original page for the description of the competition can be found here:
https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/

Part A: brief description of the organization of the project
This project designed and trained an NLP model with CNN layer and RNN layer (LSTM) embedded. Hyperparameter is carefully tuned to have a better validation accuracy.
The model could categories 5 different sentiments from phrases in datasets

Part B: the role of each file
test.tsv provides the phrases for training and validation
train.tsv contains all th phrases waiting for test to predict the sentiment
main.py includes the data preparation, model design, training and validation, testing and output prediction result function
accuracy.png shows the training and validation accuracy against epochs
test_result.csv contains the testing result
requirements.txt contains all the packages needed to be installed

sentiment_analysis.keras is the trained model which could be used directly

Part C: the packages required

run the following line to install all the packages needed
pip install -r requirements.txt