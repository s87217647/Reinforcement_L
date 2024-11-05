import functools

import gymnasium as gym
import numpy as np

from gymnasium import spaces
from gymnasium.spaces import Discrete
import pygame

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from pettingzoo.utils.env import AgentID


import pettingzoo.classic.go.go


def env(render_mode=None):
    env = MyHexGame(render_mode=render_mode)

    # wrapping?

    return env

class MyHexGame(AECEnv):
    metadata = {"name": "hex_v1"}
    def __init__(self, board_size=11, render_mode=None):
        # variables that should not be changed after the initialization
        # super.__init__()
        self.board_size = board_size
        self.possible_agents = ["black", "white"]
        self._action_spaces = {agent: Discrete(board_size ** 2) for agent in self.possible_agents}
        #todo: how to determine action value?
        #Before connecting, each move - 1? winning gives a big jackpot?
        # self._action_values = {agent: Discrete(board_size ** 2) for agent in self.possible_agents}
        # self._observation_spaces = np.zeros((board_size, board_size, 3))
        self.observation_spaces = {agent: spaces.Box(low=0, high=1, shape=(self.board_size, self.board_size, 2), dtype=bool) for agent in self.possible_agents}
        self.render_mode = render_mode
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = np.empty((self.board_size, self.board_size, 3), dtype=bool)

        # don't know if I need it
        # self.observations = {agent: None for agent in self.agents}
        self.round = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()


    @functools.lru_cache()
    # don't exactly know what it does just yet
    def action_space(self, agent):
        return self._action_spaces[agent]

    # def observation_space(self, agent):
    #     return self._observation_spaces
    #

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("calling render without specifying render mode")
            return

        else:
            board = np.empty((self.board_size, self.board_size), dtype= "str")

            #processing empty:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    black = self.state[i][j][0]
                    white = self.state[i][j][1]

                    if black:
                        board[i][j] = 'B'

                    elif white:
                        board[i][j] = 'W'

                    else:
                        board[i][j] = ' '


            print(f'\rRound {self.round}; {self.agents[0]}: horizontal; {self.agents[1]}: vertical')

            paddingCount = 0
            for row in board:
                pad = "   " * paddingCount
                print(pad + str(row))
                paddingCount += 1



    def observe(self, agent):
        #should return matching agent's obseravtion
        return self.state


    # problem now: how to match agent with the move
    # I guess I can use two iterators in the driver to make sure it always match somehow
    # reward in this case is not very important, coz I can hard code smarty's move

    def is_game_over(self) -> bool:

        return self.round * 2 >= self.board_size ** 2

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(None)

        self.round += 1

        if self.is_game_over():
            for agent in self.agents:
                self.terminations[agent] = True




        self.agent_selection = self._agent_selector.next()



        # self._cumulative_rewards[agent] = 0
        # self.state[self.agent_selection] = action


        # do the stepping automatically,
        # if self._agent_selector.is_last():
            # assert winning/ loosign status
            # pass




        if self.render_mode == "human":
            self.render()

        # return te

        return self.rewards[agent]

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))
