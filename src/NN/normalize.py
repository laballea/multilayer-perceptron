import numpy as np


class Normalize:
    def minmax(x):
        """
        normalize matrix with minmax method
        """
        if not isinstance(x, np.ndarray):
            print("normalize Invalid type.")
            return None
        result = []
        for row in x.T:
            min_r = min(row)
            max_r = max(row)
            result.append([(el - min_r) / (max_r - min_r) for el in row])
        return np.array(result).T
