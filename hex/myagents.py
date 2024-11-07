import math
import random
import myhexenv


class MyABitSmarterAgent:
    def __init__(self, statement):
        self.statement = statement
        self.log = []

    def smart_move(self, observation, action_space):
        action = self.greedy_push_right(observation, action_space)
        board_size = int(math.sqrt(action_space.n))
        r, c = action // board_size, action % board_size
        self.log.append((r + 1, chr(c + 65)))

        return action

    def reset_log(self):
        self.log = []


    def greedy_push_right(self, observation, action_space):
        # representing black go from left to right
        # pick a random left most position
        # then, push right, find out the right most, keep pushing right

        available_leftmost_col = []
        board_size = int(math.sqrt(action_space.n))

        for row in range(board_size):
            if not any(observation[row][0]):
                available_leftmost_col.append((row, 0))

        if len(available_leftmost_col) == board_size:
            chosen = random.choice(available_leftmost_col)
            chosen_row, chosen_col = chosen
            return chosen_row * board_size + chosen_col

        # now it the left most col can not be empty
        # keep track of all empty and available spaces, choosing right most one
        def hex_neighbors(r, c, board_size):
            for nr, nc in [(r - 1, c), (r - 1, c + 1), (r + 1, c), (r + 1, c - 1), (r, c - 1), (r, c + 1)]:
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    yield nr, nc

        queue = [(r, 0) for r in range(board_size) if observation[r][0][0]]
        visited = set(queue)
        adjacent_and_available = set()

        while queue:
            r, c = queue.pop(0)
            for nr, nc in hex_neighbors(r, c, board_size):
                # those neighbors are all on map neighbors
                # there are two types 1. adjacent, unvisited black stone -> queue
                # adjacent and available
                if not any(observation[nr][nc]):
                    adjacent_and_available.add((nr, nc))

                if (nr, nc) not in visited and observation[nr][nc][0]:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        right_most_row = -1
        right_most_col = -1
        for r, c in adjacent_and_available:
            if c > right_most_col:
                right_most_row = r
                right_most_col = c

        return right_most_row * board_size + right_most_col


class MyDumbAgent:
    def __init__(self, ID=None):
        self.ID = ID
        self.log = []
    def reset_log(self):
        self.log = []

    def random_action(self, observation, action_space):
        action = self._internal_random_action(observation, action_space)
        board_size = int(math.sqrt(action_space.n))
        r, c = action // board_size, action % board_size
        self.log.append((r + 1, chr(c + 65)))

        return action

    def _internal_random_action(self, observation, action_space):
        # randomly choose from available spot from observation
        board_size = int(math.sqrt(action_space.n))
        choices = []

        for action in range(action_space.n):
            row, col = myhexenv.to_coordinate(action, board_size)
            if not any(observation[row][col]):
                choices.append(action)

        if not choices:
            return None

        return random.choice(choices)
