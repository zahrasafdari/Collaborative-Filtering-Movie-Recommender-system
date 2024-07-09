import numpy as np


class Evaluation:
    @staticmethod
    def mean_manhattan_distance(prediction, true_labels):
        prediction = np.array(prediction)
        true = np.array(true_labels)
        diff = np.abs(prediction - true)
        mean = np.mean(diff)
        return mean
