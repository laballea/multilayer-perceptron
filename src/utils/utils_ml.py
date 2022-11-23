import numpy as np


def data_spliter(x, y, proportion, y_2d=False):
    """
    split data into a train set and a test set, respecting to the given proportion
    return (x_train, x_test, y_train, y_test)
    """
    if (not isinstance(x, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(proportion, float)):
        print("spliter invalid type")
        return None
    if (x.shape[0] != y.shape[0]):
        print("spliter invalid shape")
        return None
    arr = np.concatenate((x, y), axis=1)
    N = len(y)
    X = arr[:, :x.shape[1]]
    Y = arr[:, x.shape[1]]
    sample = int(proportion*N)
    np.random.shuffle(arr)
    x_train, x_test, y_train, y_test = np.array(X[:sample, :]), np.array(X[sample:, :]), np.array(Y[:sample, ]), np.array(Y[sample:, ])
    if (y_2d):
        y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    return (x_train, x_test, y_train, y_test)


def cross_validation(x, y, K, y_2d=False):
    """
    split data into n parts
    """
    if (not isinstance(x, np.ndarray) or not isinstance(x, np.ndarray)):
        print("spliter invalid type")
        return None
    if (x.shape[0] != y.shape[0]):
        print("spliter invalid shape")
        return None
    arr = np.concatenate((x, y), axis=1)
    N = len(y)
    np.random.shuffle(arr)
    for n in range(K):
        sample = int((1 / K) * N)
        test = arr[(sample * n):(sample * (n + 1))]
        train = np.concatenate([arr[0:(sample * n)], arr[(sample * (n + 1)):N]])
        x_train, y_train, x_test, y_test = train[:, :x.shape[1]], train[:, x.shape[1]], test[:, :x.shape[1]], test[:, x.shape[1]],
        
        if (y_2d):
            y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
        yield (x_train, y_train, x_test, y_test)


def add_polynomial_features(x, power):
    try:
        if (not isinstance(x, np.ndarray) or (not isinstance(power, int) and not isinstance(power, list))):
            print("Invalid type")
            return None
        if (isinstance(power, list) and len(power) != x.shape[1]):
            return None
        result = x.copy()
        if not isinstance(power, list):
            for po in range(2, power + 1):
                for col in x.T:
                    result = np.concatenate((result, (col**po).reshape(-1, 1)), axis=1)
        else:
            for col, power_el in zip(x.T, power):
                for po in range(2, power_el + 1):
                    result = np.concatenate((result, (col**po).reshape(-1, 1)), axis=1)
        return np.array(result)
    except Exception as inst:
        print(inst)
        return None


def intercept_(x):
    """
    add one columns to x
    """
    try:
        if (not isinstance(x, np.ndarray)):
            print("intercept_ invalid type")
            return None
        return np.concatenate([np.ones(len(x)).reshape(-1, 1), x], axis=1)
    except Exception as inst:
        print(inst)
        return None


def normalize(x):
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

def not_zero(x):
    eps = 1e-15
    x[x <= 0] = eps
    return x
