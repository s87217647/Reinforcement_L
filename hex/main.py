import myhexenv
from myagents import MyDumbAgent, MyABitSmarterAgent

if __name__ == '__main__':

    env = myhexenv.env(board_size=11, render_mode="human")
    env.reset()

    dummy1 = MyDumbAgent("dummy1")
    dummy2 = MyDumbAgent("dummy2")
    smarty = MyABitSmarterAgent("smarty pantsy")


    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
            # env.reset()

        else:
            if agent == 'black':  # Assign strategy for player 1
                action = smarty.smart_move(observation, env.action_space(agent))
            elif agent == 'white':  # Assign strategy for player 2
                action = dummy2.random_action(observation, env.action_space(agent))

        # Step the environment forward with the chosen action
        env.step(action)

    env.close()

    print("This is the end, hold your breath and count to ten")