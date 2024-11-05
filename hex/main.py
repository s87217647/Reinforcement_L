import itertools

import gymnasium
import myhexenv
from myagents import myDumbAgent, mySmartAgent

import itertools

if __name__ == '__main__':
    print("Hi World")

    env = myhexenv.env(render_mode="human")
    env.reset()

    dummy = myDumbAgent
    dummy2 = myDumbAgent
    smarty = mySmartAgent

    agents = (myDumbAgent("dummy1"), myDumbAgent("dummy2"))

    # for agent in itertools.cycle(agents):
    #     print(agent.ID)



    # for agent in env.agent_iter():
    #         env.step("s")
    #         print(agent)

    for agent, env_agent in zip (itertools.cycle(agents), env.agent_iter()):
        action = agent.random_action(env.action_space(env_agent), env.observe(env_agent))
        env.step(5)
        print(agent.ID, env_agent)


    # env.state[0][0][0] = True
    # env.step(5)
    # env.state[0][1][1] = True
    # env.step(5)


    # env.step(5)
    # env.step(5)
    # env.step(5)
