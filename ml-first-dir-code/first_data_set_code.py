import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("/Users/balajiwalker/Desktop/for mac/python project/ml_project/ml_package/csv_file/train.csv")
test = pd.read_csv("/Users/balajiwalker/Desktop/for mac/python project/ml_project/ml_package/csv_file/test.csv")

# Add these lines to turn off the

# train = pd.io.parsers.read_csv("train.csv")
# test = pd.io.parsers.read_csv('test.csv')

train = train.dropna(axis=0)
train_Y = train.Survived
train_predictor_columns = ['Pclass', 'Sex', 'Age', 'Fare']
train_X = train[train_predictor_columns]
test_X = test[train_predictor_columns]

# label encode the categorical values and convert them to numbers
le = LabelEncoder()
le.fit(train_X['Sex'].astype(str))
train_X['Sex'] = le.transform(train_X['Sex'].astype(str))
test_X['Sex'] = le.transform(test_X['Sex'].astype(str))

le.fit(train_X['Pclass'].astype(str))
train_X['Pclass'] = le.transform(train_X['Pclass'].astype(str))
test_X['Pclass'] = le.transform(test_X['Pclass'].astype(str))

# train the model
my_model = RandomForestRegressor()
my_model.fit(train_X, train_Y)

# fill the missing values in test data
for col in train_predictor_columns:
    mean_col = sum(train_X[col]) / len(train_X[col])
    test_X[col] = test_X[col].fillna(mean_col)

predictions = my_model.predict(test_X)
print(predictions)