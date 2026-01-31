import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

rating_df = pd.read_csv('ratings.csv')  
rating_df.head()

movies_df = pd.read_csv('movies.csv')
movies_df.head()
movies_df['genres'] = movies_df['genres'].str.split('|')
movies_df['title']=movies_df.title.str.extract('(\d\d\d\d)',expand=False)
movies_df['year']= movies_df.year.str.extract('(\d\d\d\d)',expand=False)    
movies_df.head()
movies_df['title']=movies_df.title.apply(lambda x: str(x).zfill(4))
movies_df.head()
movies_copy=movies_df.copy()
for index , row in movies_copy.iterrows():
    for genre in row['genres']:
        movies_df.at[index,genre]=1
movies_df.head()
movies_copy=movies_copy.fillna(0)
movies_copy.head()
rating_df.head()
rating_df=rating_df.drop(['timestamp'], axis=1)
rating_df.head()
rating_df=rating_df.dropna()
rating_df.head()
rating_df=rating_df.drop_duplicates()
rating_df.head()
user_input=[
      {'title' : 'Grand Slam', 'rating' : 5.6},
              {'title' : 'Zero', 'rating' : 7},
              {'title' : 'Jumanji', 'rating' : 8.5},
              {'title' : 'Toy Story', 'rating' : 4.5}
]
movies_input=pd.DataFrame(user_input)
movies_input.head()
input_id=movies_df[movies_df['title'].isin(movies_input['title'].tolist())]['movieId'].tolist()

movies_input=pd.merge(input_id,movies_input)

movies_input

movies_input=movies_input.drop(['genres','year'], axis=1)
movies_input.head()
movies_user=movies_copy[movies_copy['movieId'].isin(movies_input['movieId'].tolist())]
movies_user.head()
movies_use=movies_user.reset_index(drop=True)
UserGenreTable=movies_user.dropna(['movieId','title','genres','year'],axis=1)
UserGenreTable
Userrprofile=UserGenreTable.transpose().dot(rating_df).div(rating_df.shape[0]).fillna(0)
Userrprofile.head()
GenreTable=movies_copy.set_index(movies_copy['movieId'])
GenreTable
GenreTable=GenreTable.drop(['movieId','title','year'],axis=1)
GenreTable.head()
Recommendation_df=(GenreTable*Userrprofile).sum(axis=1)/Userrprofile.sum()
Recommendation_df.head()
Recommendation_df = Recommendation_df.sort_values(ascending=False)
Recommendation_df.head()
RecommendationTable =  movies_df.loc[movies_df['movieId'].isin(Recommendation_df.head(20).keys())]
RecommendationTable