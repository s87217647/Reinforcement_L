import math
import random

import myhexenv

from myhexenv import  to_coordinate

class mySmartAgent:
    def __init__(self):
        self.statement = "I am the smarty pantys pants"

    def make_a_move(self, observation):
        # can simply do a greedy
        pass

class myDumbAgent:
    # agent need to know the action space

    def __init__(self, ID=None):
        self.ID = ID


    def random_action(self, observation, action_space):
        #randomly choose from available spot from obseravtion
        # just be aware that picking empty space like this can be costly

        board_size = int(math.sqrt(action_space.n))
        choices = []

        for action in range(action_space.n):
            row, col = myhexenv.to_coordinate(action, board_size)
            if not any(observation[row][col]):
                choices.append(action)


        if not choices:
            return None

        return random.choice(choices)

