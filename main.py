# import numpy and pandas for data analysis
import os

import numpy as np
import pandas as pd

# import matplotlib and seaborn for visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# import files to be used for analysis
actors = pd.read_csv("/Users/rosepower/Desktop/UCD Data Analytics Course/Project/Data sets/Movie_Actors.csv")
genres = pd.read_csv("/Users/rosepower/Desktop/UCD Data Analytics Course/Project/Data sets/Movie_Genres.csv")
movies = pd.read_csv("/Users/rosepower/Desktop/UCD Data Analytics Course/Project/Data sets/Movie_Movies.csv")
writer = pd.read_csv("/Users/rosepower/Desktop/UCD Data Analytics Course/Project/Data sets/Movie_Writer.csv")

# set path for storing plots
plotfilename = '/Users/rosepower/Desktop/UCD Data Analytics Course/Project/'

# print first few lines to understand structure of files and use .info to know more about the columns of each file
print("Data from actors file:")
print(actors.head(n=2))
print(actors.info())
print(actors.shape)

print("Data from genres file:")
print(genres.head(n=2))
print(genres.info())
print(genres.shape)

print("Data from movies file:")
print(movies.head(n=2))
print(movies.info())
print(movies.shape)

print("Data from writer file:")
print(writer.head(n=2))
print(writer.info())
print(writer.shape)

# drop columns that are not required from each file
actors_cleaned = actors.drop(['Unnamed: 0'], axis=1)
genres_cleaned = genres.drop(['Unnamed: 0'], axis=1)
movies_cleaned = movies.drop(['Awards', 'DVD', 'Plot', 'Poster', 'Production', 'Rated', 'Website'], axis=1)
writer_cleaned = writer.drop(['Unnamed: 0'], axis=1)

# check that expected columns are dropped by checking the info again of the cleaned actors and movies
print(actors_cleaned.info())
print(movies_cleaned.info())

# Check for duplicates in each file
print("Number of duplicates in actors file is:" + str(actors_cleaned.duplicated().sum()))
print("Number of duplicates in genres file is:" + str(genres_cleaned.duplicated().sum()))
print("Number of duplicates in movies file is:" + str(movies_cleaned.duplicated().sum()))
print("Number of duplicates in writers file is:" + str(writer_cleaned.duplicated().sum()))


# drop duplicate rows from ratings and writers
writer_cleaned = writer_cleaned.drop_duplicates(subset='imdbID')

# Check for duplicates in each file

print("After drop_duplicates, the number of duplicates in writers file is:" + str(writer_cleaned.duplicated().sum()))


# merge files to create one dataframe with all info
movies_data = movies_cleaned.merge(writer_cleaned, on='imdbID', how='left')\
    .merge(genres_cleaned, on='imdbID')\
    .merge(actors_cleaned, on='imdbID')

print(movies_data.head())
print(movies_data.info())
print(movies_data.describe())

# Year in movies_data has Dtype object; identify what data types are in the year column
# The following function checks if the year is an integer and if not store in a dictionary for cleaning

year_data_types = {}
for year in movies_data['Year']:
    if type(year) != int:
        invalidType = type(year).__name__
        if invalidType in year_data_types.keys():
            if year not in year_data_types[invalidType]:
                year_data_types[invalidType].append(year)
        else:
            year_data_types[invalidType] = []
            year_data_types[invalidType].append(year)

print("The invalid Year types are: " + str(year_data_types.keys()))

# Check format of float type and how many there are
print(year_data_types['float'][0:10])
print(year_data_types['float'][-10:-1])
print(len(year_data_types['float']))

# Check format of string type and how many there are
print(year_data_types['str'][0:10])
print(year_data_types['str'][-10:-1])
print(len(year_data_types['str']))

# Clean the float numbers by splitting out the .0 and the strings by removing the (-) and then converting to integers
movies_data['Year'] = movies_data['Year'].apply(lambda x : int(str(x).split(".")[0]) if type(x) == float else x)
movies_data['Year'] = movies_data['Year'].apply(lambda x : x.split("–")[0] if type(x) is str else x)
movies_data['Year'] = movies_data['Year'].apply(lambda x : int(x) if type(x) != int else x)

# Check if Year column contains all integers
for row in movies_data['Year']:
    if type(row) is not int:
        print(row)
        break

# The year column is now clean

# Check if the genre column is clean by reviewing the unique values
print("The number of unique genres is: " + str(movies_data['Genre'].nunique()))
print('The unique movie genres are:')
print(movies_data['Genre'].unique())

# Trim whitespace from the Genre strings
movies_data['Genre'] = movies_data['Genre'].apply(lambda x : x.strip())

print("Following the removal of whitespace, the number of unique genres is: " + str(movies_data['Genre'].nunique()))
print('The unique movie genres are:')
print(movies_data['Genre'].unique())

# The genre column is now clean

# Identify the number of movies that fall into each genre in the past 5 years

# Subset data to movies with imdb rating
rated_movies = movies_data[movies_data['imdbRating'].notna()]
print(rated_movies.info())

print(rated_movies['Year'].max())

# Get subset of rated_movies from last 5 years

print("max year for rated_movies is: " + str(rated_movies['Year'].max()))
last_5_years = [2018, 2017, 2016, 2015, 2014]
last_5_years_movies = rated_movies[rated_movies['Year'].isin(last_5_years)]

print(last_5_years_movies.head())
print(last_5_years_movies.info())

genres_5_years = last_5_years_movies.groupby('Genre').agg({'imdbID':'count'}).reset_index()
genres_5_years_sorted= genres_5_years.sort_values(['imdbID'],ascending=False)
genrestopten_5 = genres_5_years_sorted.head(n=10)

genres_rated_movies = rated_movies.groupby('Genre').agg({'imdbID':'count'}).reset_index()
genrestopten_rated = genres_rated_movies.sort_values(['imdbID'], ascending=False).head(n=10)

genres_all_movies = movies_data.groupby('Genre').agg({'imdbID':'count'}).reset_index()
genrestopten_all_movies = genres_all_movies.sort_values(['imdbID'], ascending=False).head(n=10)

print(genrestopten_5)
genrestopten_5.plot(kind="bar")
plt.title('Top 10 genres of movies produced in last 5 years')
plt.savefig('genrestopten_5.png')

genrestopten_rated.plot(kind='bar')
plt.title('Top 10 genres of movies produced with ratings available')
plt.savefig('genrestopten_rated.png')

print(genrestopten_all_movies)
genrestopten_all_movies.plot(kind='bar')
plt.title('Top 10 genres of all movies produced')
plt.savefig('genrestopten_allmovies.png')

# Create filter for the genres in the top 10 both overall and from the past 5 years, this is the data we wish to analyse

genrefilter = []
for genre in genrestopten_5['Genre']:
    if genre not in genrefilter:
        genrefilter.append(genre)
    else:
        genrefilter = genrefilter

for genre in genrestopten_all_movies['Genre']:
    if genre not in genrefilter:
        genrefilter.append(genre)
    else:
        genrefilter = genrefilter

print(genrefilter)

movies_data['Keep'] = movies_data['Genre'].apply(lambda genre : True if genre in genrefilter else False)

# Create a new Dataframe to work with so that the general movies_data dataframe can be used for different analysis
movies_data_analysis = movies_data[movies_data['Keep']==True]

print(movies_data_analysis.info())

# Create a data set of movies between 1988 and 2018
movies_data_analysis_30years = movies_data_analysis[(movies_data_analysis['Year'] >= 1988) & (movies_data_analysis['Year'] <= 2018)]

# Create a data set of genres by year displaying the count of movies in each genre over the past 30 years
movies_genres_by_year = movies_data_analysis_30years.groupby(['Year','Genre']).agg({'imdbID':'count'}).reset_index()
print(movies_genres_by_year)

# Plot movies_genres_by_year
fig, ax = plt.subplots()
fig.set_size_inches (20,10)
sns.lineplot(data=movies_genres_by_year, x="Year", y='imdbID', hue='Genre')
plt.show()
plt.savefig('Seaborn_plot_genre_by_year.png')

# Create a data set of genres by year displaying the mean imdbRating of each genre over the past 30 years
movies_genres_rating_by_year = movies_data_analysis_30years.groupby(['Year','Genre']).agg({'imdbRating':'mean'}).reset_index()
print(movies_genres_rating_by_year)






