import gymnasium as gym
from gymnasium.envs import box2d

from mypa3_mf_control import *



if __name__ == '__main__':
    env = gym.make('CliffWalking-v0')
    # env = gym.make("FrozenLake-v1")
    ags = {'SARSA' : SARSAAgent(env), 'Q-Learning' : QLAgent(env), 'MC' : MCCAgent(env)}

    # Run each agent once.
    all_episodes = ""
    for name, agent in ags.items():
        for i in range(1, 6):
            # agent.init_qtable(env.observation_space.n, env.action_space.n)
            agent.learn()
            epi, done = agent.best_run()
            with open("./return_table.csv", "a") as table:
                epi_str = ""
                for x in epi:
                    epi_str += f'({x[0]} {x[1]} {x[2]})'

                table.write(f'{name}, {i}, {agent.calc_return(epi, done)},{epi_str} \r')


        # print(f'{name}: Best Episode: {epi}, Return: {agent.calc_return(epi, done)}')


    # mcAgent = MCCAgent(env)

    # mcAgent.learn()
    # epi, done = mcAgent.best_run()

    # agent = ValueRLAgent(env)
    # env.action_space.sample()

    # env = gym.make('CliffWalking-v0', render_mp)

    # env = gym.make("LunarLander-v3", render_mode="human")
    # observation, info = env.reset()
    #
    # episode_over = False
    # while not episode_over:
    #     action = env.action_space.sample()  # agent policy that uses the observation and info
    #     observation, reward, terminated, truncated, info = env.step(action)
    #
    #     episode_over = terminated or truncated
    #
    # env.close()

    env = gym.make('CliffWalking-v0')
    # observation, info = env.reset()
    # # print(f'starting observation {observation}, info {info}')
    #
    # episode_over = False
    # shortestPath = "URRRRRRRRRRRD"
    # shortestPath = "0111111111112"
    #
    # count = 0
    # while not episode_over:
    #     # action = env.action_space.sample()  # agent policy that uses the observation and info
    #     action = int(shortestPath[count])
    #     count += 1
    #
    #     observation, reward, terminated, truncated, info = env.step(action)
    #
    #     episode_over = terminated or truncated
    #     print(action, observation, reward, terminated, truncated, info)
    #
    # env.close()

    # mcc = MCCAgent(env)
    # mcc.learn()
    # epi, done = mcc.best_run()
    # print(f'Best Episode: {epi}, Return: {mcc.calc_return(epi, done)}')


    # sarsa = SARSAAgent(env)
    # sarsa.learn()
    # epi, done = sarsa.best_run()
    # print(f'Best Episode: {epi}, Return: {sarsa.calc_return(epi, done)}')


    # ql = QLAgent(env)
    # ql.learn()
    # epi, done = ql.best_run()
    # print(f'Best Episode: {epi}, Return: {ql.calc_return(epi, done)}')
