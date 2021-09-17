import unittest
from ddt import ddt, data, unpack
import numpy as np
from wolfPolicyTransitionNoiseAllStates import grid_transition_noise

bounds = [15, 15]


def is_state_valid(state):
    if state[0] not in range(bounds[0]):
        return False
    if state[1] not in range(bounds[1]):
        return False
    return True


# def grid_transition_noise(s=(), a=(), A=(), is_valid=None, terminals=(), noise=0.1):

    # @ddt
    # class TestTransition(unittest.TestCase):
    #     def setUp(self):
    #         self.A = ((1, 0), (0, 1), (-1, 0), (0, -1))

    #     @data(([[2, 2], [10, 10]], True), ([[10, 23], [100, 100]], False))
    #     @unpack
    #     def testTranstionWithNoise(self, state, action, actionSpace=self.A, is_valid=is_state_valid, noise=0.1):

    #         self.assertEqual(


if __name__ == '__main__':
    # unittest.main()
    state = (0, 0)
    action = (1, 0)
    A = ((1, 0), (0, 1), (-1, 0), (0, -1))
    nextState = grid_transition_noise(state, action, A, is_state_valid, (), 0.1)
    print(nextState)
