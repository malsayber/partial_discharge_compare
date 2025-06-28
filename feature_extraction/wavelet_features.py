import math


def _haar_level(signal):
    """One level Haar transform for a sequence."""
    data = [float(x) for x in signal]
    if len(data) % 2:
        data.append(data[-1])
    cA = []
    cD = []
    for i in range(0, len(data), 2):
        avg = (data[i] + data[i + 1]) / math.sqrt(2.0)
        diff = (data[i] - data[i + 1]) / math.sqrt(2.0)
        cA.append(avg)
        cD.append(diff)
    return cA, cD


def wavelet_energy(signal, level=3):
    """Compute energies of Haar wavelet coefficients up to given level."""
    coeff = [float(x) for x in signal]
    energies = {}
    for lvl in range(1, level + 1):
        coeff, detail = _haar_level(coeff)
        energies[f'D{lvl}'] = sum(d * d for d in detail)
    energies[f'A{level}'] = sum(c * c for c in coeff)
    return energies
