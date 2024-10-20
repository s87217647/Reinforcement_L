from typing import Any
import random
import gymnasium as gym
import numpy as np


def argmax_action(d: dict[Any, float]) -> Any:
    """return a key of the maximum value in a given dictionary 

    Args:
        d (dict[Any,float]): dictionary

    Returns:
        Any: a key
    """

    max_val = max(d.values())
    keys = []

    for k in d.keys():
        if d[k] == max_val:
            keys.append(k)

    return random.choice(keys)



class ValueRLAgent():
    def __init__(self, env: gym.Env, gamma: float = 0.98, eps: float = 0.2, alpha: float = 0.02,
                 total_epi: int = 5_000) -> None:
        """initialize agent parameters
        This class will be a parent class and not be called directly.

        Args:
            env (gym.Env): gym environment
            gamma (float, optional): a discount factor. Defaults to 0.98.
            eps (float, optional): the epsilon value. Defaults to 0.2. Note: this pa uses a simple eps-greedy not decaying eps-greedy.
            alpha (float, optional): a learning rate. Defaults to 0.02.
            total_epi (int, optional): total number of episodes an agent should learn. Defaults to 5_000.
        """
        self.env = env
        self.q = self.init_qtable(env.observation_space.n, env.action_space.n)
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.total_epi = total_epi

    def init_qtable(self, n_states: int, n_actions: int, init_val: float = 0.0) -> dict[int, dict[int, float]]:
        """initialize the q table (dictionary indexed by s, a) with a given init_value

        Args:
            n_states (int, optional): the number of states. Defaults to int.
            n_actions (int, optional): the number of actions. Defaults to int.
            init_val (float, optional): all q(s,a) should be set to this value. Defaults to 0.0.

        Returns:
            dict[int,dict[int,float]]: q table (q[s][a] -> q-value)
        """
        # does the state starts with 0 or 1?
        q_table = dict()
        for s in range(n_states):
            actions_for_s = dict()
            for a in range(n_actions):
                actions_for_s[a] = init_val

            q_table[s] = actions_for_s

        return q_table

    def eps_greedy(self, state: int, exploration: bool = True) -> int:
        """epsilon greedy algorithm to return an action

        Args:
            state (int): state
            exploration (bool, optional): explore based on the epsilon value if True; take the greedy action by the current self.q if False. Defaults to True.

        Returns:
            int: action
        """
        opt_action = argmax_action(self.q[state])
        other_actions = list(self.q[state].keys())
        other_actions.remove(opt_action)


        if not exploration:
            return opt_action

        # max action's prob: 1 - epsilon + ep/n
        if random.random() < 1 - self.eps + self.eps / len(self.q[state]):
            return opt_action
        else:
            return random.choice(other_actions)




    def choose_action(self, ss: int) -> int:
        """a helper function to specify a exploration policy
        If you want to use the eps_greedy, call the eps_greedy function in this function and return the action.

        Args:
            ss (int): state

        Returns:
            int: action
        """
        # why ss here is a state?
        return self.eps_greedy(ss, True)

    def best_run(self, max_steps: int = 100) -> tuple[list[tuple[int, int, float]], bool]:
        """After the learning, an optimal episode (based on the latest self.q) needs to be generated for evaluation. From the initial state, always take the greedily best action until it reaches a goal.

        Args:
            max_steps (int, optional): Terminate the episode generation if the agent cannot reach the goal after max_steps. One step is (s,a,r) Defaults to 100.

        Returns:
            tuple[
                list[tuple[int,int,float]],: An episode [(s1,a1,r1), (s2,a2,r2), ...]
                bool: done - True if the episode reaches a goal, False if it hits max_steps.
            ]
        """
        episode = list()
        done = False

        state, info = self.env.reset()

        over = False
        while not over:
            if max_steps == 0:
                break

            max_steps -= 1

            max_val_action = argmax_action(self.q[state])
            observation, reward, terminated, truncated, info = self.env.step(max_val_action)
            episode.append((state, max_val_action, reward))
            state = observation

            done = terminated
            over = truncated or terminated

        return (episode, done)

    def calc_return(self, episode: list[tuple[int, int, float]], done=False) -> float:
        """Given an episode, calculate the return value. An episode is in this format: [(s1,a1,r1), (s2,a2,r2), ...].

        Args:
            episode (list[tuple[int,int,float]]): An episode [(s1,a1,r1), (s2,a2,r2), ...]
            done (bool, optional): True if the episode reaches a goal, False if it does not. Defaults to False.

        Returns:
            float: the return value. None if done is False.
        """

        if not done:
            return None


        G = 0
        episode.reverse()
        for step in episode:
            G = self.gamma * G + step[2]

        return G



class MCCAgent(ValueRLAgent):
    def episode_generation(self, max_steps):
        episode = []
        state, observation = self.env.reset()

        for step in range(max_steps):
            action = self.choose_action(state)
            observation, reward, terminated, truncated, info = self.env.step(action)
            episode.append((state, action, reward))
            state = observation

            if terminated or truncated:
                break

        return episode
    def learn(self) -> None:
        """Monte Carlo Control algorithm
        Update the Q table (self.q) for self.total_epi number of episodes.

        Note: When an episode is too long (> 500 for CliffWalking), you can stop the episode and update the table using the partial episode.

        The results should be reflected to its q table.
        """

        max_steps = 500
        for i in range(self.total_epi):
            episode = self.episode_generation(max_steps)
            for j in range(len(episode)):
                G = self.calc_return(episode[j:], True)

                step = episode[j]
                self.q[step[0]][step[1]] += self.alpha * (G - self.q[step[0]][step[1]])


class SARSAAgent(ValueRLAgent):
    def learn(self) -> None:
        """SARSA algorithm
        Update the Q table (self.q) for self.total_epi number of episodes.

        The results should be reflected to its q table.
        """
        for i in range(self.total_epi):
            state, info = self.env.reset()

            while True:
                action = self.choose_action(state)
                ss, reward, terminated, truncated, info = self.env.step(action)
                self.q[state][action] += self.alpha * (reward + self.gamma * self.q[ss][self.choose_action(ss)] - self.q[state][action])
                state = ss

                if terminated or truncated:
                    break




class QLAgent(SARSAAgent):
    def learn(self):
        """Q-Learning algorithm
        Update the Q table (self.q) for self.total_epi number of episodes.

        The results should be reflected to its q table.
        """
        for i in range(self.total_epi):
            state, info = self.env.reset()

            while True:
                action = self.choose_action(state)
                ss, reward, terminated, truncated, info = self.env.step(action)
                self.q[state][action] += self.alpha * (reward + self.gamma * self.q[ss][self.choose_action(ss)] - self.q[state][action])
                state = ss

                if terminated or truncated:
                    break




    def choose_action(self, ss: int) -> int:
        """
        [optional] You may want to override this method.
        """
        return argmax_action(self.q[ss])


