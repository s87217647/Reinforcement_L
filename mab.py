import random

import numpy as np
import matplotlib.pyplot as plt

def ucb1_bandit(arms, num_steps):
    """
    UCB1 algorithm for the multi-armed bandit problem.

    Args:
    arms (list): List of arms (bandit machines) with their true reward probabilities.
    num_steps (int): Number of steps (iterations) to run the algorithm.

    Returns:
    selected_arms (list): List of arms selected at each step.
    rewards (list): List of rewards obtained at each step.
    """
    num_arms = len(arms)
    num_pulls = np.zeros(num_arms)  # Number of times each arm has been pulled.
    total_rewards = np.zeros(num_arms)  # Total rewards obtained from each arm.
    selected_arms = []  # List to store the selected arms.
    rewards = []  # List to store the rewards obtained.

    for step in range(num_steps):
        pass

    return selected_arms, rewards


def epsilon_greedy_bandit(arms, num_steps, epsilon):
    """
    Epsilon-Greedy algorithm for the multi-armed bandit problem.

    Args:
    arms (list): List of arms (bandit machines) with their true reward probabilities.
    num_steps (int): Number of steps (iterations) to run the algorithm.
    epsilon (float): The exploration-exploitation trade-off parameter (0 <= epsilon <= 1).

    Returns:
    selected_arms (list): List of arms selected at each step.
    rewards (list): List of rewards obtained at each step.
    """
    num_arms = len(arms)
    num_pulls = np.zeros(num_arms)  # Number of times each arm has been pulled.
    total_rewards = np.zeros(num_arms)  # Total rewards obtained from each arm.
    selected_arms = []  # List to store the selected arms.
    rewards = []  # List to store the rewards obtained.

    def reward(p) -> int:
        if random.random() <= p:
            return 1
        return 0

    for step in range(num_steps):
        if random.random() <= epsilon:
            # random choice, explore
            choice = random.randint(0, num_arms - 1)
            num_pulls[choice] += 1
            selected_arms.append(choice)
            r = reward(arms[choice])
            total_rewards[choice] += r
            rewards.append(r)
        else:
            choice = 0
            for i in range(num_arms):
                if total_rewards[i] > total_rewards[choice]:
                    choice = i

            selected_arms.append(choice)
            num_pulls[choice] += 1
            r = reward(arms[choice])
            total_rewards[choice] += r
            rewards.append(r)



    return selected_arms, rewards

if __name__ == '__main__':
    arms = [0.9, 0.8, 0.7, 0.5, 0.4]

    selected_arm, rewards = epsilon_greedy_bandit(arms, 9999, 0.1)

    print(selected_arm)
    print(rewards)