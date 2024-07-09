import numpy as np
from sklearn.neighbors import NearestNeighbors


class ItemBased:

    def __init__(self, data, n_neighbors):
        self.data = data
        self.n_neighbors = n_neighbors
        self.movies_mean_score = self.get_movies_mean_score()
        self.rating_matrix = self.get_rating_matrix()
        self.knn_model = self.build_knn_model()

    def get_movies_mean_score(self):
        # calculate mean of scores for each movie
        n_movies = self.data.max_movies + 1
        each_movie_sum_scores = np.zeros(n_movies)
        each_movie_num_users_seen = np.zeros(n_movies)
        for idx, row in self.data.df_ratings_train.iterrows():
            each_movie_sum_scores[row["MovieID"]] += row["Rating"]
            each_movie_num_users_seen[row["MovieID"]] += 1
        # we find mean by doing sum/n_occurrence (divisions by 0 are considered to be 0)
        movie_mean_scores = np.divide(each_movie_sum_scores, each_movie_num_users_seen,
                                      out=np.zeros_like(each_movie_sum_scores), where=each_movie_num_users_seen != 0)
        return movie_mean_scores

    def get_rating_matrix(self):
        """
        :return: returns filled matrix  -->  [movie_id, user_id] = rating
        """
        n_movies = self.data.max_movies + 1
        n_users = self.data.max_users + 1
        rating_matrix = np.zeros((n_movies, n_users))
        for idx, row in self.data.df_ratings_train.iterrows():
            rating_matrix[row["MovieID"], row["UserID"]] = row["Rating"] - self.movies_mean_score[row["MovieID"]]
        return rating_matrix

    def build_knn_model(self):
        neigh = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine')
        neigh.fit(self.rating_matrix)
        return neigh

    def predict(self, df_ratings_test):
        predictions = []
        for idx, row in df_ratings_test.iterrows():
            prediction = self.predict_one(row["UserID"], row["MovieID"])
            predictions.append(prediction)
        return predictions

    def predict_one(self, user_id, movie_id):
        """
        :return: returns the predicted score of a movie that a user has watched
        """
        knn_similarities, knn_movie_ids = self.knn_model.kneighbors(self.rating_matrix[movie_id].reshape(1, -1))
        knn_similarities = knn_similarities[0].tolist()
        knn_movie_ids = knn_movie_ids[0].tolist()
        df_ratings_train = self.data.df_ratings_train
        correlation_sum = similarity_sum = 0

        for knn_movie_id, similarity in zip(knn_movie_ids, knn_similarities):
            rating = df_ratings_train["Rating"].loc[(df_ratings_train['MovieID'] == knn_movie_id) & (
                    df_ratings_train["UserID"] == user_id)]
            if rating.empty:
                continue
            rating = rating.values[0]  # just get the rating value instead of getting it as a pd.series
            correlation_sum += rating * similarity
            similarity_sum += similarity
        if similarity_sum == 0:
            return 3.6
        prediction = correlation_sum / similarity_sum
        return prediction
