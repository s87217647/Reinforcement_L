from mymdp import MDP
from mypa2_dp import ValueAgent, PIAgent, VIAgent


if __name__ == '__main__':

    mdp = MDP("./mdp1.json")

    zero_val_states = {'0': 0, '1': 0, '2': 0}

    # va = ValueAgent(mdp)
    # va.init_random_policy()
    # print(va.computeq_fromv(zero_val_states))
    # print(va.pi)
    # va.greedy_policy_improvement(zero_val_states)
    # print(va.pi)
    #al_states))


    pia = PIAgent(mdp)
    pia.policy_iteration()
    for x in pia.v_update_history:
        print(x)

    #
    # via = VIAgent(mdp)
    # print(via.value_iteration())
    # for x in via.v_update_history:
    #     print(x)


    print("This is the end, hold your breath and cout to ten")



