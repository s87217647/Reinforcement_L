import itertools

import gymnasium
import myhexenv
from myagents import myDumbAgent, mySmartAgent

import itertools

if __name__ == '__main__':

    env = myhexenv.env(board_size=8, render_mode="human")
    env.reset()


    dummy1 = myDumbAgent("dummy1")
    dummy2 = myDumbAgent("dummy2")



    # #todo: make sure step is AEC
    # for agent, env_agent in zip(itertools.cycle(agents), env.agent_iter()):
    #     action = agent.random_action(env.action_space(env_agent), env.observe(env_agent))
    #     env.step(action)
    #     print(agent.ID, env_agent)

    # while not env.is_game_over():
    #     agent = env.agent_selection
    #     observation, reward, termination, truncation, info = env.last()
    #
    #     if termination or truncation:
    #         action = None  # Skip if the episode is over for this agent
    #     else:
    #         if agent == 'black':  # Assign strategy for player 1
    #             action = dummy1.random_action(observation, env.action_space(agent))
    #         elif agent == 'white':  # Assign strategy for player 2
    #             action = dummy2.random_action(observation, env.action_space(agent))
    #
    #     # Step the environment forward with the chosen action
    #     env.step(action)

    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            break
            # action = None  # Skip if the episode is over for this agent

        else:
            if agent == 'black':  # Assign strategy for player 1
                action = dummy1.random_action(observation, env.action_space(agent))
            elif agent == 'white':  # Assign strategy for player 2
                action = dummy2.random_action(observation, env.action_space(agent))

        # Step the environment forward with the chosen action
        env.step(action)


    env.close()






    print("This is the end, hold your breath and count to ten")