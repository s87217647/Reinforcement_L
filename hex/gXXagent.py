import random


class GXXAgent:
    def __init__(self, env):
        self.env = env

    def select_action(self, observation, reward, termination, truncation, info):
        valid_action = []
        for i in range(len(info["action_mask"])):
            if info["action_mask"][i]:
                valid_action.append(i)

        choice = random.choice(valid_action)
        return choice
