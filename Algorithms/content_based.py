from Utils.data_manager import DataManager
from Utils.similarity_metrics import SimilarityMetrics
import numpy as np


class ContentBased:

    def __init__(self, data, n_neighbors):
        self.data = data
        self.n_neighbors = n_neighbors
        self.movie_genre_matrix = self.get_movie_genre_matrix()
        self.movies_similarity_matrix = self.get_movies_similarity_matrix()

    def get_movie_genre_matrix(self):
        """
        :return: returns filled matrix  -->  [movie_id, genre] = rating
        """
        n_movies = self.data.max_movies
        n_genres = len(DataManager.genres)
        movie_genre_matrix = np.zeros((n_movies + 1, n_genres))  # [movie_id, genre] = rating
        for idx, row in self.data.df_movies.iterrows():
            number_of_genres = len(DataManager.genres)  # number_of_genres is 18 for this dataset
            for genre_idx in range(number_of_genres):
                if row[genre_idx]:
                    movie_genre_matrix[row["MovieID"], genre_idx] = 1
        return movie_genre_matrix

    def get_movies_similarity_matrix(self):
        """
        :return: returns pre calculated movie to movie similarities  -->  [movie_id, movie_id] = similarity
        """
        n_movies = self.data.max_movies
        movies_similarity_matrix = np.zeros((n_movies + 1, n_movies + 1))  # [movie_id, movie_id] = similarity
        for movie_id1 in range(n_movies):
            for movie_id2 in range(0, movie_id1):
                movies_similarity_matrix[movie_id1, movie_id2] = movies_similarity_matrix[
                    movie_id1, movie_id2] = self.get_movies_similarities(movie_id1, movie_id2)
        return movies_similarity_matrix

    def get_movies_similarities(self, movie_id1, movie_id2):
        movie_genre_vector1 = np.array(self.movie_genre_matrix[movie_id1])
        movie_genre_vector2 = np.array(self.movie_genre_matrix[movie_id2])
        similarity = SimilarityMetrics.cosine_similarity(movie_genre_vector1, movie_genre_vector2)
        return similarity

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
        knn_similarity_id_rating = self.knn(user_id, movie_id)
        correlation_sum = similarity_sum = 0
        for one_similarity_id_rating in knn_similarity_id_rating:
            similarity = one_similarity_id_rating[0]
            rating = one_similarity_id_rating[2]
            correlation_sum += rating * similarity
            similarity_sum += similarity
        if similarity_sum == 0:
            return 3.6
        prediction = correlation_sum / similarity_sum
        return prediction

    def knn(self, user_id, movie_id):
        """
        :return: returns a 2D list in size K rows (K nearest ones) and 3 columns (similarity, movie_id, rating).
        each row is a movie's description:
        1) similarity: the similarity between the movie_id we got from the function and a movie this user has
        watched before
        2) movie_id: the id of the aforementioned movie
        3) rating: the rating the user has given to the aforementioned movie
        -> final dimension: [K, 3]
        """
        list_similarity_and_userid = []
        df_ratings_train = self.data.df_ratings_train
        df_this_user_movies = df_ratings_train[df_ratings_train["UserID"] == user_id]
        for index, row in df_this_user_movies.iterrows():
            if row['MovieID'] != movie_id:
                similarity = self.movies_similarity_matrix[movie_id, row["MovieID"]]
                list_similarity_and_userid.append([similarity, row["MovieID"], row["Rating"]])

        knn_similarity_id_rating = sorted(list_similarity_and_userid, reverse=True)[
                                   :min(self.n_neighbors, len(list_similarity_and_userid))]
        return knn_similarity_id_rating
