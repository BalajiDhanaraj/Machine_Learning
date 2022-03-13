import pandas as pd

file_path = "/Volumes/Macintosh HD/For Mac/python project/Machine_Learning/csv-data set/melb_data.csv"

home_data = pd.read_csv(file_path)

""" describe() function is used to show the datas"""
print(home_data.describe())








