import pandas as pd

"""
What is pandas ---> its one of the python library for data analysis.
"""

## Creating data using DataFrame

""" A DataFrame is a table, its contains an array of individual entries """

print(pd.DataFrame(
    {
    'Yes':[50,32],
     'No':[232,3]
    }
   )
)

print(pd.DataFrame(
    {
     'Title':['My name is walker'],
      'Learn':['Pandas'],
    }
  )
)

## we using the pd.dataframe() constructor to generate these DataFrame

## creating index in the dataframe

print(
    pd.DataFrame(
        {
            'Title':['My name is steve'],
            'work':['My work is QA']
        },
        index=['Product A:', 'Product B:']
    )
)

## Series --> a series by contrast is a sequence of data value --> its more like list
## Dataframe is a table , and Series is a list

print(pd.Series([1,3,2,3]))


print(
    pd.Series(
        [3,4,5],
        index=['2010 Sales','2016 Sales','2017 Sales'],
        name='Product A'
    )
)

## importing the csv data file

review = pd.read_csv("/Volumes/Macintosh HD/For Mac/python project/Machine_Learning/csv-data set/winemag-data_first150k.csv")

print(review.head(5))

print(pd.set_option('max_rows',5))

##












