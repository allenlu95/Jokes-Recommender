
import graphlab
import pandas as pd
import numpy as np
import hsl


#%cd /Users/allen/Desktop/Galvanize/dsi-recommender-case-study
df_ratings = pd.read_table('/Users/allen/desktop/galvanize/dsi-recommender-case-study/data/ratings.dat')

#added 10 and raised it to the fourth to make differences more apparent
df_ratings['rating'] = (df_ratings['rating']+10)**4
df_jokes = pd.read_table('/Users/allen/desktop/galvanize/dsi-recommender-case-study/data/jokes.dat')
df_ratings.head()

#Sparse Matrix Factorization, with side data in the form of tfidf vectorizers
ratings_gl = graphlab.SFrame(df_ratings)
item_data = graphlab.SFrame(hsl.nmf_df)
recommender = graphlab.recommender.factorization_recommender.create(ratings_gl,user_id = 'user_id', item_id = 'joke_id', target = 'rating', solver = 'als', max_iterations = 100, num_factors=7, item_data = item_data)

#Predict on test set
df = pd.read_csv('/Users/allen/desktop/galvanize/dsi-recommender-case-study/data/test_ratings.csv')
points = graphlab.SFrame({'user_id': df['user_id'].values, 'joke_id': df['joke_id'].values})
p = recommender.predict(points)
predictions = p

df['rating'] = np.array(predictions)
df.to_csv('predictions.csv')
