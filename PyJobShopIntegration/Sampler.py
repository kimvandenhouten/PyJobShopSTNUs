import numpy as np
from scipy.stats import randint


class DiscreteRVSampler:
    def __init__(self):
        pass

    def sample(self):
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


