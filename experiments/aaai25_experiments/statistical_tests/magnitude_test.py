import numpy as np
import scipy

"""
This Script implements the magnitude test or t-test for two population means (method of paired comparisons) which is 
test 10 such as described in "100 Statistical Tests - 3rd Edition" by Gopal K. Kanji. Note that the critical region 
should be obtained from the standard normal approximation. Note that additionally buid in a normalization step.
"""


def normalize_data(obs_1, obs_2):
    assert len(obs_1) == len(obs_2)
    n = len(obs_1)
    norm_1 = []
    norm_2 = []
    for i in range(n):
        mean = (obs_1[i] + obs_2[i]) / 2
        if mean > 0:
            norm_1.append(obs_1[i] / mean)
            norm_2.append(obs_2[i] / mean)
        else:
            norm_1.append(1)
            norm_2.append(1)
    return norm_1, norm_2


def magnitude_test(obs_1, obs_2, normalize = True):
    assert len(obs_1) == len(obs_2)
    n = len(obs_1)
    if normalize:
        obs_1, obs_2 = normalize_data(obs_1, obs_2)

    result = scipy.stats.ttest_rel(obs_1, obs_2, alternative='two-sided')

    return result, np.mean(obs_1), np.mean(obs_2), n