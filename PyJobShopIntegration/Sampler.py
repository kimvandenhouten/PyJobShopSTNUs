import numpy as np
from scipy.stats import randint
from scipy.stats import binom


class DiscreteRVSampler:
    def __init__(self):
        pass

    def sample(self, num_samples=1):
        raise NotImplementedError

    def get_bounds(self):
        raise NotImplementedError


class DiscreteUniformSampler(DiscreteRVSampler):
    def __init__(self, lower_bounds, upper_bounds):
        super(DiscreteUniformSampler, self).__init__()
        """
        Initializes the sampler using scipy.stats.randint.

        :param lower_bounds: List or array of lower bounds for each dimension.
        :param upper_bounds: List or array of upper bounds for each dimension.
                             (Exclusive upper bounds, as required by scipy.stats.randint)
        """

        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)

        if len(self.lower_bounds) != len(self.upper_bounds):
            raise ValueError("Lower and upper bounds must have the same length.")

        if np.any(self.upper_bounds < self.lower_bounds):
            raise ValueError("Each upper bound must be strictly greater than the corresponding lower bound.")

        # Create scipy.stats.randint distributions for each dimension
        # Make upper bounds inclusive
        self.distributions = [randint(low, high) for low, high in zip(self.lower_bounds, self.upper_bounds+1)]

    def sample(self, num_samples=1):
        """
        Generates samples from the discrete uniform distribution using scipy.stats.

        :param num_samples: Number of samples to generate.
        :return: NumPy array of shape (num_samples, num_dimensions)
        """
        samples = np.column_stack([dist.rvs(size=num_samples) for dist in self.distributions])
        if num_samples == 1:
            return samples[0]
        else:
            return samples

    def get_bounds(self):
        return self.lower_bounds, self.upper_bounds

    def get_quantile(self, quantile):
        if all(self.lower_bounds == self.upper_bounds):
            quantile = self.lower_bounds
        else:
            quantile = [int(self.lower_bounds[k] + quantile * (self.upper_bounds[k] - self.lower_bounds[k] + 1) - 1) for k in range(len(self.lower_bounds))]

        return quantile

class DiscreteBinomialSampler(DiscreteRVSampler):
    def __init__(self, lower_bounds, upper_bounds, p=0.5):
        """
        Initializes a binomial sampler for each dimension.

        :param lower_bounds: List or array of lower bounds for each dimension.
        :param upper_bounds: List or array of upper bounds for each dimension (inclusive).
        :param p: Probability of success for the binomial distribution.
        """
        super().__init__()
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.p = p

        if len(self.lower_bounds) != len(self.upper_bounds):
            raise ValueError("Lower and upper bounds must have the same length.")

        if np.any(self.upper_bounds < self.lower_bounds):
            raise ValueError("Each upper bound must be greater than or equal to the corresponding lower bound.")

        # Number of trials is upper - lower
        self.n_trials = self.upper_bounds - self.lower_bounds
        self.distributions = [
            binom(n=n, p=self.p) for n in self.n_trials
        ]

    def sample(self, num_samples=1):
        """
        Generates samples from the binomial distributions and shifts them to respect lower bounds.

        :param num_samples: Number of samples to generate.
        :return: NumPy array of shape (num_samples, num_dimensions)
        """
        binomial_samples = np.column_stack([dist.rvs(size=num_samples) for dist in self.distributions])
        shifted_samples = binomial_samples + self.lower_bounds
        if num_samples == 1:
            return shifted_samples[0]
        else:
            return shifted_samples

    def get_bounds(self):
        return self.lower_bounds, self.upper_bounds

    def get_quantile(self, quantile):
        if not 0 <= quantile <= 1:
            raise ValueError("Quantile must be between 0 and 1.")

        quantiles = np.array([dist.ppf(quantile) for dist in self.distributions], dtype=int)
        return quantiles + self.lower_bounds


