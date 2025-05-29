import numpy as np

from Sampler import DiscreteUniformSampler


class FJSP():

    def __init__(self, model):
        self.model = model

    def get_durations(self):
        return [mode.duration for mode in self.model.modes]


    def duration_distributions(self, noise_factor):
        lower_bound, upper_bound = self.get_bounds(self.get_durations(), noise_factor)
        duration_distributions = DiscreteUniformSampler(
            lower_bounds=lower_bound,
            upper_bounds=upper_bound
        )
        return duration_distributions

    def get_bounds(self, durations, noise_factor):
        lb = []
        ub = []
        for duration in durations:
            if duration == 0:
                lb.append(0)
                ub.append(0)
            else:
                lower_bound = int(max(1, duration - noise_factor * np.sqrt(duration)))
                upper_bound = int(duration + noise_factor * np.sqrt(duration))
                if lower_bound == upper_bound:
                    upper_bound += 1
                lb.append(lower_bound)
                ub.append(upper_bound)
        return lb, ub