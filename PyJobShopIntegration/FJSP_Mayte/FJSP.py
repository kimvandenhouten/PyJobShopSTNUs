from collections import defaultdict

import numpy as np
from pyjobshop import Mode, Model

from Sampler import DiscreteUniformSampler


class FJSP():

    def __init__(self, model):
        self.model = model

    def get_durations(self):
        return [mode.duration for mode in self.model.modes]

    def model_new_durations(self, new_durations):

        data = self.model.data()

        # 2) Sanity‐check
        if len(new_durations) != len(data.modes):
            raise ValueError(
                f"Length mismatch: model has {len(data.modes)} modes, "
                f"but you passed {len(new_durations)} durations."
            )

        # 3) Build a new list of Mode objects, copying everything except duration
        new_modes = [
            Mode(
                task=mode.task,
                resources=mode.resources,
                duration=new_d,
                demands=mode.demands
            )
            for mode, new_d in zip(data.modes, new_durations)
        ]

        new_data = data.replace(modes=new_modes)
        return Model.from_data(new_data)

    def duration_distributions(self, noise_factor):
        lower_bound, upper_bound = self.get_bounds(noise_factor)
        duration_distributions = DiscreteUniformSampler(
            lower_bounds=lower_bound,
            upper_bounds=upper_bound
        )
        return duration_distributions

    def get_bounds(self, noise_factor):
        lb = []
        ub = []
        for duration in self.get_durations():
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
