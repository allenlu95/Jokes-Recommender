# Jokes-Recommender
Recommend funny jokes

This projects works with text data in the form of jokes, from the jester dataset, http://eigentaste.berkeley.edu/dataset/. The project is divided into 2 sections, finding latent features and recommendation.

# Latent Features
In this section I take in the jokes, and then clean each one by tokenizing, lemmatizing and removing stop words. I then pass the cleaned dataset into a tf-idf vectorizer to get a large matrix with the importance of each word in each joke. Finally I perform non-matrix factorization on the tf-idf features, and try out different features and decide on 10 latent features. With more time I will take the SVD of the tf-idf vector and perform the elbow method to find how many features can give us 90% power.

# Recommending
In this section I used graphlab to peform sparse matrix factorization with item features (extracted from the tf-idf matrix earlier). I include the elbow method; however, there doesn't seem to be a significant cut-off point. I settle on 10 features to match the latent features selected. The idea behind the error metric I used, is the order of which 5 jokes a user most enjoyed does not matter, but the getting those 5 jokes does. So in this example we want to find the top 5% of jokes each user rated. Since it is the top 5% jokes, a higher score is preferred. Luckily the actual data was provided to us and the ceiling to our scores had an average of 15.9 out of 20. Our model got us at 13.1 out of 20.
