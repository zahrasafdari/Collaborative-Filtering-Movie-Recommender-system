import numpy as np
import math


class SimilarityMetrics:

    @staticmethod
    def cosine_similarity(a, b):
        """
        Takes 2 vectors a, b and returns the cosine similarity according
        to the definition of the dot product
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        # print(dot_product, norm_a, norm_b, a, b)
        if math.isnan(norm_a) or math.isnan(norm_b) or norm_b == 0 or norm_a == 0:
            return 0
        return dot_product / (norm_a * norm_b)
