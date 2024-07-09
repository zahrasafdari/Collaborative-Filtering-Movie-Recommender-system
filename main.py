import os
from Utils.data_manager import DataManager
from Utils.evaluation import Evaluation
from recommender_system import RecommenderSystem

def main():
    # Define the paths for the data files
    current_path = os.getcwd()
    users_path = os.path.join(current_path, "Data/ml-1m/users.dat")
    movies_path = os.path.join(current_path, "Data/ml-1m/movies.dat")
    ratings_path = os.path.join(current_path, "Data/ml-1m/ratings.dat")

    # Load the data
    data = DataManager(
        path_users=users_path, 
        path_movies=movies_path, 
        path_ratings=ratings_path, 
        train_test_split=0.05, 
        max_users=1000, 
        max_movies=1000
    )

    # Initialize recommender systems
    user_based_rs = RecommenderSystem(data, n_neighbors=50, method_name="user_based")
    item_based_rs = RecommenderSystem(data, n_neighbors=50, method_name="item_based")
    content_based_rs = RecommenderSystem(data, n_neighbors=50, method_name="content_based")

    # Generate predictions for the test data
    user_based_predictions = user_based_rs.predict(data.df_ratings_test)
    item_based_predictions = item_based_rs.predict(data.df_ratings_test)
    content_based_predictions = content_based_rs.predict(data.df_ratings_test)

    # Retrieve true labels
    true_ratings = data.df_ratings_test["Rating"].tolist()

    # Evaluate predictions
    user_based_score = Evaluation.mean_manhattan_distance(prediction=user_based_predictions, true_labels=true_ratings)
    item_based_score = Evaluation.mean_manhattan_distance(prediction=item_based_predictions, true_labels=true_ratings)
    content_based_score = Evaluation.mean_manhattan_distance(prediction=content_based_predictions, true_labels=true_ratings)

    # Print the evaluation results
    print("Our classifiers results:")
    print(f"Loss in 'user_based' recommender system = {user_based_score}")
    print(f"Loss in 'item_based' recommender system = {item_based_score}")
    print(f"Loss in 'content_based' recommender system = {content_based_score}\n")

    # Calculate and evaluate the greedy baseline
    mean_rating = sum(true_ratings) / len(true_ratings)
    print(f'The mean value of scores is: {mean_rating:.2f}.')
    greedy_predictions = [mean_rating] * len(true_ratings)
    greedy_score = Evaluation.mean_manhattan_distance(prediction=greedy_predictions, true_labels=true_ratings)
    print(f"Loss in 'greedy' recommender system = {greedy_score}")

if __name__ == '__main__':
    main()

