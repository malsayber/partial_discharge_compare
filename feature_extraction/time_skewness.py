import math


def time_skewness(signal):
    """Return the skewness of a 1-D sequence."""
    data = [float(x) for x in signal]
    n = len(data)
    mean = sum(data) / n
    var = sum((x - mean) ** 2 for x in data) / n
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    third = sum((x - mean) ** 3 for x in data) / n
    return third / (std ** 3)
