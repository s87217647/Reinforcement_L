import functools

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector


def env(board_size=11, render_mode=None):
    env = MyHexGame(board_size=board_size, render_mode=render_mode)

    # wrappings

    return env


class MyHexGame(AECEnv):
    metadata = {"name": "hex_v1"}

    def __init__(self, board_size=11, render_mode=None):
        # variables that should not be changed after the initialization
        board_size = 20 if board_size > 20 else board_size
        board_size = 5 if board_size < 5 else board_size

        self.board_size = board_size
        self.possible_agents = ["black", "white"]
        self._action_spaces = {agent: Discrete(board_size ** 2) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(self.board_size, self.board_size, 2), dtype=bool) for agent in
            self.possible_agents}
        self.render_mode = render_mode


    def reset(
            self,
            seed: int | None = None,
            options: dict | None = None,
    ) -> None:
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = np.empty((self.board_size, self.board_size, 2), dtype=bool)
        # so far, observation is the same as the state
        # self.observations = {agent: None for agent in self.agents}
        self.agents_to_board = {"black": 0, "white": 1}
        self.move_count = 0

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
            board = np.empty((self.board_size, self.board_size), dtype="str")

            # processing empty:
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

            print(f'\nmove count:{self.move_count}')

            paddingCount = 0
            for row in board:
                pad = "   " * paddingCount
                print(pad + str(row))
                paddingCount += 1

    def observe(self, agent):
        # black is horizontal & first board
        return self.state

    def neighbors(self, r, c):
        for nr, nc in [(r - 1, c), (r - 1, c + 1), (r + 1, c), (r + 1, c - 1), (r, c - 1), (r, c + 1)]:
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.state[nr][nc][self.agents_to_board[self.agent_selection]]:
                yield nr, nc

    def white_vertical_connected(self) -> bool:
        # white is vertical
        rows, cols = self.board_size, self.board_size
        # Start BFS from the top row (adjust if left-to-right connection is needed)
        queue = [(0, c) for c in range(cols) if self.state[0][c][1]]
        visited = set(queue)

        while queue:
            r, c = queue.pop(0)

            # If we reach the bottom row, the board is connected
            if r == rows - 1:
                return True

            for nr, nc in self.neighbors(r, c):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return False

    def black_horizontal_connected(self) -> bool:
        rows, cols = self.board_size, self.board_size
        # Start BFS from the top row (adjust if left-to-right connection is needed)
        queue = [(r, 0) for r in range(rows) if self.state[r][0][0]]
        visited = set(queue)

        while queue:
            r, c = queue.pop(0)

            if c == cols - 1:
                return True

            for nr, nc in self.neighbors(r, c):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return False

    def got_a_winner(self):
        if self.agent_selection == "black":
            return self.black_horizontal_connected()
        else:
            return self.white_vertical_connected()

    def is_full(self) -> bool:
        # connection exists
        full = True
        for row in range(self.board_size):
            for col in range(self.board_size):
                cell = self.state[row][col]
                if not (cell[0] or cell[1]):
                    full = False

        return full

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        opponent = "white" if agent == "black" else "black"

        col, row = to_coordinate(action, self.board_size)
        target_location = self.state[col][row]

        if not any(target_location): #no conflicts
            target_location[self.agents_to_board[agent]] = True
        else:
            if self.move_count == 1: # pie rule
                target_location[self.agents_to_board[agent]] = True
                target_location[self.agents_to_board[opponent]] = False

            else:# conflicts shouldn't exist
                gym.logger.warn(f"{agent} conflicts at {row}, {col}")

        self.rewards[agent] = -1

        if self.got_a_winner():
            print(f'{agent} won')
            self.rewards[agent] = 100
            self.terminations = {agent: True for agent in self.agents}

        if self.is_full():
            self.terminations = {agent: True for agent in self.agents}

        self._cumulative_rewards[agent] += self.rewards[agent]
        self.move_count += 1
        self.infos[agent]["moves"] = self.move_count

        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()

    def close(self):
        # do nothing, don't need to release graphic interface or hardware
        pass


def to_coordinate(action: int, board_size: int) -> (int, int):
    col = action // board_size
    row = action % board_size

    return col, row
