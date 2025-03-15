import numpy as np
import scipy
from scipy.stats import binomtest, binom

n = 1000
p = 0.45
p_zero = 0.5
z_critical = 1.96
k = int(p * n)

# Manual calculation
def z_func(p, p_zero, n):
    a = np.abs(p-p_zero) - 1 / (2 * n)
    b = (p_zero * (1 - p_zero)) / n
    z = a / (b ** 0.5)
    return z


z_manual = z_func(p, p_zero, n)
p_value_manual = scipy.stats.norm.sf(abs(z_manual)) * 2

p_val_check = scipy.stats.norm.sf(abs(1.96))

print(f'p_val check {p_val_check}')
print(f'p_val check two-sided {p_val_check * 2} \n')

if -1.96 < z_manual < 1.96:
    null = True
    print(f'We accept the null hypothesis')
else:
    null = False
    print(f'We reject the null hypothesis')


print(f'z={z_manual} is z_value according to manual calculation from statistics book')
print(f'p={p} is test statistic according to manual calculation from statistics book')
print(f'p={p_value_manual} is p-value according to manual calculation from statistics book\n')

# SciPy calculation
from scipy.stats import binomtest, binom

result = binomtest(k, n, p_zero, "greater")
p_value_scipy = result.pvalue
statistic = result.statistic
a = np.abs(p-p_zero) - 1 / (2 * n)
b = (p_zero * (1 - p_zero)) / n
z_scipy = a / (b ** 0.5)
print(f'z={z_scipy} is z_value according to scipy calculation')
print(f'p={statistic} is test statistic according to scipy')
print(f'p={p_value_scipy} is p-value according to scipy')