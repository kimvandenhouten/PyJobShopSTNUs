import numpy as np
import scipy

"""
This Script implements the proportion test (binomial test), that is also test 4 such as described in "100 Statistical
Tests - 3rd Edition" by Gopal K. Kanji. Note that the critical region should be obtained from the standard normal 
approximation.
"""

def z_func(n, k, p_zero):
    p = k / n
    a = np.abs(p-p_zero) - 1 / (2 * n)
    b = (p_zero * (1 - p_zero)) / n
    z = a / (b ** 0.5)
    return z


def proportion_test(n, k, p_zero, z_crit):
    p = k / n
    z_manual = z_func(n, k, p_zero)
    p_value_manual = scipy.stats.norm.sf(abs(z_manual)) * 2
    if -z_crit < z_manual < z_crit:
        null = True
        print(f'We accept the null hypothesis')
    else:
        null = False
        print(f'We reject the null hypothesis')

    return z_manual, p_value_manual, null, p