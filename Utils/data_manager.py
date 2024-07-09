import pandas as pd
import math


class DataManager:
    genres = {"Action": 0, "Adventure": 1, "Animation": 2, "Children's": 3, "Comedy": 4, "Crime": 5, "Documentary": 6,
              "Drama": 7, "Fantasy": 8, "Film-Noir": 9, "Horror": 10, "Musical": 11, "Mystery": 12, "Romance": 13,
              "Sci-Fi": 14, "Thriller": 15, "War": 16, "Western": 17}

    def __init__(self, path_users, path_movies, path_ratings, train_test_split=0.1, max_users=1000, max_movies=1000):
        self.max_users = max_users
        self.max_movies = max_movies
        self.path_users = path_users
        self.path_movies = path_movies
        self.path_ratings = path_ratings
        # get the data in DataManager
        self.df_users = self.get_users()
        self.df_movies = self.get_movies()
        self.df_ratings_train, self.df_ratings_test = self.get_ratings(train_test_split=train_test_split)

    def get_users(self):
        df = pd.read_csv(self.path_users,
                         engine='python',
                         sep="::",
                         names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],encoding='latin1')
        df = df[df["UserID"] <= self.max_users]
        df = df.drop(columns='Zip-code')
        df = pd.get_dummies(df, columns=["Gender", "Age", "Occupation"])
        return df

    def get_movies(self):
        df = pd.read_csv(self.path_movies,
                         engine='python',
                         sep="::",
                         names=["MovieID", "Title", "Genres"], encoding='latin1')
        df = df[df["MovieID"] <= self.max_movies]
        df = df.drop(columns='Title')
        df = self.split_movies_genres_in_df(df)
        df = df.drop(columns="Genres")
        return df

    def split_movies_genres_in_df(self, df):
        """
        example:   turns  df["Genres"]="Drama|Action|Comedy"  ---to-->   df["7"] = df["4"] = df["0"] = 1
        :param df:
        :return:
        """
        for genre_key, genre_val in DataManager.genres.items():
            df[genre_val] = 0

        for index, row in df.iterrows():
            new_df_row = row
            for existing_genre in row["Genres"].split("|"):
                new_df_row[self.genres[existing_genre]] = 1
            df.iloc[index] = new_df_row  # replace the old row with the new one
        return df

    def get_ratings(self, train_test_split):
        df = pd.read_csv(self.path_ratings,
                         engine='python',
                         sep="::",
                         names=["UserID", "MovieID", "Rating", "Timestamp"], encoding='latin1')
        df = df[(df["MovieID"] <= self.max_movies) & (df["UserID"] <= self.max_users)]
        df = df.drop(columns='Timestamp')
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
        # split train_test
        split_index = math.floor(len(df) * train_test_split)
        df_test = df.iloc[:split_index]
        df_train = df.iloc[split_index + 1:]
        return df_train, df_test
