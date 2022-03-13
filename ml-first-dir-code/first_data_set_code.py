import pandas as pd

file_path = "/Volumes/Macintosh HD/For Mac/python project/Machine_Learning/csv-data set/melb_data.csv"

home_data = pd.read_csv(file_path)

""" describe() function is used to show the datas"""
print(home_data.describe())
"""Selecting Data for Modeling
Your dataset had too many variables to wrap your head around, or even to print out nicely. How can you pare down this overwhelming amount of data to something you can understand?

We'll start by picking a few variables using our intuition. Later courses will show you statistical techniques to automatically prioritize variables.

To choose variables/columns, we'll need to see a list of all columns in the dataset. That is done with the columns property of the DataFrame (the bottom line of code below)."""
print(home_data.columns)

# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.
# Your Iowa data doesn't have missing values in the columns you use.
# So we will take the simplest option for now, and drop houses from our data.
# Don't worry about this much for now, though the code is:

melbourne_data = home_data.dropna(axis=0)
print(melbourne_data)

## we have to predict the target , so predtiction target is called by "y"
## the dot notation to select the column we want to predict, which is called the "prediction target".
y = melbourne_data.Price

"""Choosing "Features"
The columns that are inputted into our model (and later used to make predictions) are called "features." In our case, those would be the columns used to determine the home price. Sometimes, you will use all columns except the target as features. Other times you'll be better off with fewer features.

For now, we'll build a model with only a few features. Later on you'll see how to iterate and compare models built with different features.

We select multiple features by providing a list of column names inside brackets. Each item in that list should be a string (with quotes)."""

## we have chooosed the prediction target as y, so we have to give input data is called as "Features" it is have column to are going to predict

melbourne_feature = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

## above columns we are going to give as input

X = melbourne_data[melbourne_feature] ### melbourne data have all data set so in list we going to give few data colunmn

print(X.describe())

print(X.head())
print(X.tail())
print("------------------------------------------------------------")


""" Building YOur Model 

the step to building and using a model

Define ---> what type of model will it be? A decision tree? some other type of model?
           some other parameter of the model type are specifed too
Fit -------> Capture patttern from provided data . This is the heart of modeling 
              capture pattern mean divided the group of names like
              finding the house price by no of rooms, no of toilet, no of living rooms,
              size of the rooms, so capture pattern mean dividing the data into the valued groups to predict
Predict ---> Just what is sounds like

Evaluate ---> Determine how accurate the models prediction are
"""

from sklearn.tree import DecisionTreeRegressor as DTR
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DTR(random_state=1)
## Fiting the captured pattern
melbourne_model.fit(X,y)
### Next step to predict
print("Making predictions for the following 5 house")
print(X.head(5))
print("The predictions are")
print(melbourne_model.predict(X.head(5)))

print("------------------------------------------------------------")

"""" Prediction 2 """










































