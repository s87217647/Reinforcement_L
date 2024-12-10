import random

from g05agent import G05Agent
from gXXagent import GXXAgent
from ourhexenv import OurHexGame

env = OurHexGame(board_size=11, sparse_flag=False)
env.reset()


def runner():
    # player 1
    dump_agent = GXXAgent(env)
    # player 2
    smartAgent = G05Agent(env, training_mode=True)

    smart_agent_player_id = random.choice(env.agents)

    done = False
    while not done:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                if smartAgent.training_mode:
                    print("reset")
                    env.reset()
                    smart_agent_player_id = random.choice(env.agents)

                else:
                    done = True
                    break


            if agent == smart_agent_player_id:
                action = smartAgent.select_action(observation, reward, termination, truncation, info)

                if not smartAgent.training_mode:
                    print(f"smart agent action is {action}")
                    env.step(action)

            else:
                action = dump_agent.select_action(observation, reward, termination, truncation, info)
                env.step(action)

            env.render()


if __name__ == '__main__':
    # train first, then run

    runner()

    print("This is the end, hold your breath and count to ten")
