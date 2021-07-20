# import numpy and pandas for data analysis
import os
import numpy as np
import pandas as pd

# import matplotlib and seaborn for visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# import files to be used for analysis

genres = pd.read_csv("/Users/rosepower/Desktop/UCD Data Analytics Course/Project/Data sets/Movie_Genres.csv")
movies = pd.read_csv("/Users/rosepower/Desktop/UCD Data Analytics Course/Project/Data sets/Movie_Movies.csv")
writers = pd.read_csv("/Users/rosepower/Desktop/UCD Data Analytics Course/Project/Data sets/Movie_Writer.csv")

# print first few lines to understand structure of files and use .info to know more about the columns of each file
print("Data from genres file:")
print(genres.head(n=2))
print(genres.info())
print(genres.shape)

print("Data from movies file:")
print(movies.head(n=2))
print(movies.info())
print(movies.shape)

print("Data from writers file:")
print(writers.head(n=2))
print(writers.info())
print(writers.shape)

# drop columns that are not required from each file
genres_cleaned = genres.drop(['Unnamed: 0'], axis=1)
movies_cleaned = movies.drop(['Awards', 'DVD', 'Plot', 'Poster', 'Production', 'Rated', 'Website'], axis=1)
writers_cleaned = writers.drop(['Unnamed: 0'], axis=1)

# check that expected columns are dropped by checking the info again of movies and genre
print(movies_cleaned.info())
print(genres_cleaned.info())
print(writers_cleaned.info())

# Check for duplicates in each file
print("Number of duplicates in genres file is:" + str(genres_cleaned.duplicated().sum()))
print("Number of duplicates in movies file is:" + str(movies_cleaned.duplicated().sum()))
print("Number of duplicates in writers file is:" + str(writers_cleaned.duplicated().sum()))

# Drop duplicates from the writers file
writers_drop_dup = writers_cleaned.drop_duplicates(keep='first')
print("Number of duplicates in writers file is:" + str(writers_drop_dup.duplicated().sum()))
print(writers_drop_dup.info())

# merge files to create one dataframe with all info
movies_data = movies_cleaned.merge(genres_cleaned, on='imdbID')

# print information about the new dataframe
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

# Check for null values in the key colums: Year, imdbRating and Genre
print(movies_data.isna().sum())

# Find what % of data is null for each column
print(movies_data.isna().sum() * 100 / len(movies_data))

# Find what % of data is null for each genre
null_by_genre = movies_data.groupby('Genre').agg({'imdbRating': lambda x: x.isnull().sum()}).reset_index()
num_movies_by_genre = movies_data.groupby('Genre').agg({'imdbID': 'count'}).reset_index()
null_by_genre_prec = null_by_genre.merge(num_movies_by_genre, on='Genre')
null_by_genre_prec['Percent'] = null_by_genre_prec['imdbRating'] * 100 / null_by_genre_prec['imdbID']
print(null_by_genre_prec.sort_values(['imdbID'], ascending = False))

# The data is now clean and ready for analysis.

# Subset data to movies with imdb rating
rated_movies = movies_data[movies_data['imdbRating'].notna()]
print(rated_movies.info())

# Set values for last 5 years which can be reused
last_5_years = [2017, 2016, 2015, 2014, 2013]

# Get subset of rated_movies from last 5 years
last_5_years_movies = rated_movies[rated_movies['Year'].isin(last_5_years)]

# Check the info related to the movies from the last 5 years
print(last_5_years_movies.info())

# Identify top 10 genres from rated_movies in past 5 years based on number of movies
genres_5_years = last_5_years_movies.groupby('Genre').agg({'imdbID':'count'}).reset_index()
genrestopten_5= genres_5_years.sort_values(['imdbID'],ascending=False).head(n=10)

# Identify top 10 genres from all rated_movies years based on number of movies
genres_rated_movies = rated_movies.groupby('Genre').agg({'imdbID':'count'}).reset_index()
genrestopten_rated = genres_rated_movies.sort_values(['imdbID'], ascending=False).head(n=10)

# Identify top 10 genres from all movies (movies_data) based on number of movies
genres_all_movies = movies_data.groupby('Genre').agg({'imdbID':'count'}).reset_index()
genrestopten_all_movies = genres_all_movies.sort_values(['imdbID'], ascending=False).head(n=10)

# Print the dataframe
print(genrestopten_5)

# Generate a bar graph of the top ten genres of rated movies from past 5 years
genrestopten_5.plot(kind='bar', x='Genre', y='imdbID')
plt.xticks(rotation=45)
plt.title('Top 10 genres of movies produced in last 5 years')
plt.savefig('genrestopten_5.png')

# Print the list of top ten genres of all rated movies
print(genrestopten_rated)

# Generate a bar graph of the top ten genres of all rated movies
genrestopten_rated.plot(kind='bar', x='Genre', y='imdbID')
plt.xticks(rotation=45)
plt.title('Top 10 genres of movies produced with ratings available')
plt.savefig('genrestopten_rated.png')

# Print the list of the top ten genres of all rated movies
print(genrestopten_all_movies)

# Generate a bar graph of the top ten genres of all rated movies
genrestopten_all_movies.plot(kind='bar', x='Genre', y='imdbID')
plt.xticks(rotation=45)
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

# Print the genre filter list
print(genrefilter)

# Add a new column to rated_movies to show which movies to keep
# And add True of False based on if genre of movie is in the genre filter
rated_movies['Keep'] = rated_movies['Genre'].apply(lambda genre : True if genre in genrefilter else False)

# Create a new Dataframe of rated movies with the Genres we wish to work with
# to work with so that the general rated_movies dataframe can be used for different analysis
rated_movies_analysis = rated_movies.loc[rated_movies['Keep']==True]

print(rated_movies_analysis.info())

# Create a data set of movies between 1987 and 2017
rated_movies_analysis_30years = rated_movies_analysis[(rated_movies_analysis['Year'] >= 1987) & (rated_movies_analysis['Year'] <= 2017)]

# Create a data set of genres by year displaying the count of movies in each genre over the past 30 years
movies_genres_by_year = rated_movies_analysis_30years.groupby(['Year','Genre']).agg({'imdbID':'count'}).reset_index()
print(movies_genres_by_year)

# Plot movies_genres_by_year
fig, ax = plt.subplots()
fig.set_size_inches (20,10)
sns.set_theme(style="whitegrid")
sns.lineplot(data=movies_genres_by_year, x="Year", y='imdbID', hue='Genre', palette='tab10')
plt.title("Number of rated movies by Genre")
plt.savefig('Seaborn_plot_genre_by_year.png')
plt.show()

# Create a data set of genres by year displaying the mean imdbRating of each genre over the past 30 years
movies_genres_rating_by_year = rated_movies_analysis_30years.groupby(['Year','Genre']).agg({'imdbRating':'mean'}).reset_index()
print(movies_genres_rating_by_year)

# Plot movies_genres_rating_by_year
fig, ax = plt.subplots()
fig.set_size_inches (20,10)
sns.set_theme(style="whitegrid")
sns.lineplot(data=movies_genres_rating_by_year, x="Year", y='imdbRating', hue='Genre', palette='tab10')
plt.title("Average ratings by Genre")
plt.savefig('Seaborn_plot_genre_rating_by_year.png')
plt.show()