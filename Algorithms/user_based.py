import numpy as np
from sklearn.neighbors import NearestNeighbors


class UserBased:

    def __init__(self, data, n_neighbors):
        self.data = data
        self.n_neighbors = n_neighbors
        self.users_mean_score = self.get_users_mean_score()
        self.rating_matrix = self.get_rating_matrix()
        self.knn_model = self.build_knn_model()

    def get_users_mean_score(self):
        # calculate mean of scores for each user
        n_users = self.data.max_users + 1
        each_user_sum_scores = np.zeros(n_users)
        each_user_num_movies_seen = np.zeros(n_users)
        for idx, row in self.data.df_ratings_train.iterrows():
            each_user_sum_scores[row["UserID"]] += row["Rating"]
            each_user_num_movies_seen[row["UserID"]] += 1
        # we find mean by doing sum/n_occurrence (divisions by 0 are considered to be 0)
        users_mean_scores = np.divide(each_user_sum_scores, each_user_num_movies_seen,
                                      out=np.zeros_like(each_user_sum_scores), where=each_user_num_movies_seen != 0)
        return users_mean_scores

    def get_rating_matrix(self):
        """
        :return: returns filled matrix  -->  [user_id, movie_id] = rating
        """
        n_movies = self.data.max_movies + 1
        n_users = self.data.max_users + 1
        rating_matrix = np.zeros((n_users, n_movies))
        for idx, row in self.data.df_ratings_train.iterrows():
            rating_matrix[row["UserID"], row["MovieID"]] = row["Rating"] - self.users_mean_score[row["UserID"]]
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
        knn_similarities, knn_user_ids = self.knn_model.kneighbors(self.rating_matrix[user_id].reshape(1, -1))
        knn_similarities = knn_similarities[0].tolist()
        knn_user_ids = knn_user_ids[0].tolist()
        df_ratings_train = self.data.df_ratings_train
        correlation_sum = similarity_sum = 0

        for knn_user_id, similarity in zip(knn_user_ids, knn_similarities):
            rating = df_ratings_train["Rating"].loc[(df_ratings_train['UserID'] == knn_user_id) & (
                    df_ratings_train["MovieID"] == movie_id)]
            if rating.empty:
                continue
            rating = rating.values[0]  # just get the rating value instead of getting it as a pd.series
            correlation_sum += rating * similarity
            similarity_sum += similarity
        if similarity_sum == 0:
            return 3.6
        prediction = correlation_sum / similarity_sum
        return prediction
