''' This is content based movie recommendation
	system program in python. In this program we
	will be using a 100k movie dataset for recommending movies

	***Steps to keep in mind***
	1. Include necessary modules
	2. Read and Clean the DataSet 
	3. Now do Exploratory Data Analysis
	4. Gettin Movie Recommendation
	5. create a predict function
'''
import numpy as np 
import pandas as pd 
import warnings
import matplotlib.pyplot as plt 
import seaborn as sns

warnings.filterwarnings('ignore')

#get the dataset
column_names = ["user_id","item_id","rating","timestamp"]
df = pd.read_csv('ml-100k/u.data',sep='\t',names= column_names)
#print(df.head())
#print(df.shape)
#print(df['user_id'].nunique())
#print(df['item_id'].nunique())

movies_titles = pd.read_csv("ml-100k/u.item",sep="\|",header= None)

movies_titles = movies_titles[[0,1]]
movies_titles.columns = ["item_id","title"]

#print(movies_titles.head())

df = pd.merge(df,movies_titles, on = "item_id")

#print(df.tail())
'''Till now we have read our data now we are doning some *EXPLORATORY DATA ANALYSIS* using seaborn and matplotlib'''
#print(df.groupby('title').mean()['rating'].sort_values(ascending =False).head())
#print(df.groupby('title').count()['rating'].sort_values(ascending = False).head())

ratings = pd.DataFrame(df.groupby('title').mean()['rating'])

#print(ratings.tail(n=10))

ratings['num of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])
ratings = ratings.sort_values(by = 'rating',ascending=False)
#print(ratings.head())

''''plt.figure(figsize = (10,6))
plt.hist(ratings['num of ratings'], bins= 70)
plt.show()

plt.hist(ratings['rating'],bins=70)
plt.show()

sns.jointplot(x= 'rating',y='num of ratings',data=ratings,alpha=0.5)
plt.show()'''

'''Now in the above we have seen many graphs, that are showing how many number of peoples rated a movie and the rating'''
''' Now we work on creating a movie recommendation
	1. We create a movie matrix with index = user_id and columns = title.
	2. Now we pick a movie title.
	3. We find correaltion of that movie with other movies in the matrix and create a dataframe of it:
		* We drop the value which is NaN
		* We sort the values 
		* we join the ratings['num of ratings'] column to our correlation dataframe.
	4. We consider the rating which is rated by at least 100 people and sort the dataframe acc. to correlation
	5. In the end we return our prediction
'''

moviemat = df.pivot_table(index="user_id",columns = "title",values='rating')
#print(moviemat)
ratings = ratings.sort_values("num of ratings",ascending=False)
#print(ratings)
'''
	****This is the working of our function. In this we have taken movie=  star wars (1977) and made prediction on that.****
	starwar_user_rating = moviemat['Star Wars (1977)']
	print(starwar_user_rating)
	similar_to_starwars = moviemat.corrwith(starwar_user_rating)
	print(similar_to_starwars)
	corr_starwars = pd.DataFrame(similar_to_starwars,columns=["correlation"])
	print(corr_starwars)
	corr_starwars.dropna(inplace=True)
	corr_starwars = corr_starwars.sort_values("correlation",ascending=False)
	print(corr_starwars)
	corr_starwars = corr_starwars.join(ratings['num of ratings'])
	print(corr_starwars.head())
	corr_starwars = corr_starwars[corr_starwars['num of ratings']>100].sort_values('correlation',ascending=False)
	print(corr_starwars.head())

'''
def predict_movies(movie_name):
	movie_user_rating = moviemat[movie_name]
	similar_to_movie = moviemat.corrwith(movie_user_rating)
	corr_movie = pd.DataFrame(similar_to_movie,columns=['correlation'])
	corr_movie.dropna(inplace=True)
	corr_movie = corr_movie.join(ratings['num of ratings'])
	prediction = corr_movie[corr_movie['num of ratings']>100].sort_values("correlation",ascending=False)
	return prediction

predictions = predict_movies("Titanic (1997)")
print(predictions.head())