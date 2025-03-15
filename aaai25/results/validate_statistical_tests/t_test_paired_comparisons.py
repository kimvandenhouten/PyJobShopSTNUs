import numpy as np

obs_1 = [30.02, 29.99, 30.11, 29.97, 30.01, 29.99]
obs_2 = [29.89, 29.93, 29.72, 29.98, 30.02, 29.98]
print(obs_1)
print(obs_2)

# Manual calculations
assert len(obs_1) == len(obs_2)

n = len(obs_1)

diff = [obs_1[i] - obs_2[i] for i in range(n)]
mean_diff = np.mean(diff)

delta_squared = [((diff[i] - mean_diff) ** 2) / (n - 1) for i in range(n)]
s_squared = np.sum(delta_squared)

t = (np.mean(obs_1) - np.mean(obs_2)) / (np.sqrt(s_squared) / (n**0.5))
print(f't value is {t} according to manual calculation')

# SciPy calculations

import scipy

result = scipy.stats.ttest_rel(obs_1, obs_2, alternative='two-sided')
print(result)


# Now with normalized data
# We must first normalize the data
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


# Manual calculations
print("\n")
print(norm_1)
print(norm_2)
diff = [norm_1[i] - norm_2[i] for i in range(n)]
mean_diff = np.mean(diff)

delta_squared = [((diff[i] - mean_diff) ** 2) / (n - 1) for i in range(n)]
s_squared = np.sum(delta_squared)

t = (np.mean(norm_1) - np.mean(norm_2)) / (np.sqrt(s_squared) / (n**0.5))
print(f't value is {t} according to manual calculation')

# SciPy calculations

import scipy

result = scipy.stats.ttest_rel(norm_1, norm_2, alternative='two-sided')
print(result)

