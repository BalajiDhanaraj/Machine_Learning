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

# print(pd.set_option('max_rows',5))

## Accessing the column using the name

print(review['country'].head(5))

## Accessing the data using the index [0]

print(review['country'][1])
print("---------------------------------------------",end=" ")
""" 
Indexing in pandas
accessing the data using pandas with method of --> loc and iloc,
Both loc and iloc are row-first, column-second, but in python - column-first , row-seconds
"""

# Index - based selection #

print(review.iloc[0]) # row

# to display the column in iloc

print(review.iloc[:,1])

""" first index row and second index is column"""
print(review.iloc[0:3,0:3])

## using the negative numbers to retrived the data from the last

print(review.iloc[-5:-10,:3])

""" Label-based selection -->loc,  """

print(review.loc[0,'country'])


""" Manipulating the index -- index we use is not immutable, we can manipulate the index in any way we see fit"""

# print(review.set_index("title"))

## conditional selection

print(review.country=='Italy')

print(review.loc[(review.country == 'Italy') & (review.points >= 90)])

## isin --- data is in a list of values,

print(review.loc[review.country.isin(['Italy','France'])])

""" isnull --->not null mean is empty (NaN """

print(review.loc[review.price.isnull()])

print(review.loc[review.price.notnull()])

"""Assigning data """

ass = review['critic'] = 'everyone'

print(ass)

## describe function is used to show the over all data.
des = review.describe()
print(des)


cols = ['country', 'province', 'region_1', 'region_2']
indices = [0, 1, 10, 100]
df = review.loc[indices, cols]

print(df)

top_oceania_wines = review.loc[(review.country.isin(['Australia','New Zealand'])) & (review.points>=95)]

print(top_oceania_wines)


""" Summary function and maps """

# summary function
#Pandas provides many simple "summary functions"
# (not an official name) which restructure the data in some useful way.
# For example, consider the describe() method: will help us to view all the basic details
# count         103727
# unique            19
# top       Roger Voss
# freq           25514
# Name: taster_name, dtype: object


print(review.points.describe())

# we can also view only particular summary of the dataframe

print(review.points.mean())

# "unique()" function can help us to view the unique list of values in dataframe

print(review.taster_name.unique())











