import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column_name= ['userid','item_id','rating','timestamp']
df = pd.read_csv('ratings.csv',names=column_name)
df.head()
movies_title=pd.read_csv('movies.csv')
movies_title.head()
data=pd.merge(df,movies_title,on='item_id')
data.head()
data.groupby('title')['rating'].mean().sort_values(ascending=False).head()
data.groupby('title')['rating'].count().sort_values(ascending=False).head()
ratings=pd.DataFrame(data.groupby('title')['rating'].mean())
ratings['num of ratings']=pd.DataFrame(data.groupby('title')['rating'].count())
ratings.head()

sns.set_style('white')
plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)
plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)
moviemat=data.pivot_table(index='userid',columns='title',values='rating')
moviemat.head()
ratings.sort_values('num of ratings',ascending=False).head(10)
starwars_user_ratings=moviemat['Star Wars (1977)']
liarliar_user_ratings=moviemat['Liar Liar (1997)']
starwars_user_ratings.head()
similarity_starwars=moviemat.corrwith(starwars_user_ratings)
similarity_liarliar=moviemat.corrwith(liarliar_user_ratings)
similarity_starwars.head()
top_10_starwars=similarity_starwars.sort_values(ascending=False).head(10)
corr_starwars=pd.DataFrame(similarity_starwars,columns=['Correlation'])
corr_starwars.head()
corr_starwars.sort_values('Correlation',ascending=False).head(10)
corr_starwars=corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head(10)
corrliarliar=similarity_liarliar.sort_values(ascending=False).head(10)
corrliarliar=pd.DataFrame(similarity_liarliar,columns=['Correlation'])
corrliarliar.dropna(inplace=True)
corrliarliar=corrliarliar.join(ratings['num of ratings'])
corrliarliar[corrliarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head(10)
