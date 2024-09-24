from mymdp import MDP
from mypa2_dp import ValueAgent, PIAgent, VIAgent


if __name__ == '__main__':
    mdp = MDP("./mdp1.json")


    pia = PIAgent(mdp)
    print(pia.policy_iteration())
    print(pia.v_update_history[-1])



    via = VIAgent(mdp)
    print(via.value_iteration())
    print(via.v_update_history[-1])

    with open("./pi_history.log", 'w') as f:
        for h in pia.v_update_history:
            f.write(str(h) + '\r')

    with open("./vi_hisotry.log", 'w') as f:
        for h in via.v_update_history:
            f.write(str(h) + '\r')


    print("This is the end, hold your breath and cout to ten")



