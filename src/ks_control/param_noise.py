import numpy as np
from math import sqrt

"""
From OpenAI Baselines:
https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
"""
class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=1, desired_action_stddev=0.05, adaptation_coefficient=1.005):
        """
        Note that initial_stddev and current_stddev refer to std of parameter noise, 
        but desired_action_stddev refers to (as name notes) desired std in action space
        """
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = np.mean(np.square(actions1-actions2), axis=0)
    mean_diff = np.mean(diff)
    dist = sqrt(mean_diff)
    return dist
