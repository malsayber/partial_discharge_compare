import math


def time_kurtosis(signal):
    """Return the kurtosis of a 1-D sequence (not excess)."""
    data = [float(x) for x in signal]
    n = len(data)
    mean = sum(data) / n
    var = sum((x - mean) ** 2 for x in data) / n
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    fourth = sum((x - mean) ** 4 for x in data) / n
    return fourth / (std ** 4)
