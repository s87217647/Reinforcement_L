import copy
import random

from mymdp import MDP


class ValueAgent:
    """Value-based Agent template (Used as a parent class for VIAgent and PIAgent)
    An agent should maintain:
    - q table (dict[state,dict[action,q-value]])
    - v table (dict[state,v-value])
    - policy table (dict[state,dict[action,probability]])
    - mdp (An MDP instance)
    - v_update_history (list of the v tables): [Grading purpose only] Every time when you update the v table, you need to append the v table to this list. (include the initial value)
    """

    def __init__(self, mdp: MDP, conv_thresh: float = 0.000001) -> None:
        """Initialization

        Args:
            mdp (MDP): An MDP instance
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.            
        """
        self.q = dict()
        self.v = dict()
        self.pi = dict()
        self.mdp = mdp
        self.thresh = conv_thresh
        self.v_update_history = list()

    def init_random_policy(self):
        """Initialize the policy function with equally distributed random probability.

        When n actions are available at state s, the probability of choosing an action should be 1/n.
        """
        # policy table(dict[state, dict[action, probability]])
        for s in self.mdp.states():
            policy = dict()
            actions = self.mdp.actions(s)

            for a in actions:
                policy[a] = 1 / len(actions)

            self.pi[s] = policy

    def init_random_state(self):
        # Initialize state value
        for s in self.mdp.states():
            # actions exists, not the terminal state
            if self.mdp.actions(s):
                self.v[s] = random.random() - 50
            else:
                self.v[s] = 0

    def computeq_fromv(self, v: dict[str, float]) -> dict[str, dict[str, float]]:
        """Given a state-value table, compute the action-state values.
        For deterministic actions, q(s,a) = E[r] + v(s'). Check the lecture slides.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            dict[str,dict[str,float]]: a q value table {state:{action:q-value}}
        """
        actionValTable = dict()

        for s in self.mdp.states():
            action_to_val = dict()

            for a in self.mdp.actions(s):
                action_val = 0
                for (sstate, p) in self.mdp.T(s, a):
                    # print(s, a, "expected reward = ", p * (self.mdp.R(s, a, sstate) + self.mdp.gamma * v[sstate]))
                    action_val += p * (self.mdp.R(s, a, sstate) + self.mdp.gamma * v[sstate])

                action_to_val[a] = action_val

            actionValTable[s] = action_to_val

        self.q = actionValTable
        return actionValTable

    def greedy_policy_improvement(self, v: dict[str, float]) -> dict[str, dict[str, float]]:
        """Greedy policy improvement algorithm. Given a state-value table, update the policy pi.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        action_val_table = self.computeq_fromv(v)

        for s in self.mdp.states():
            if max(action_val_table[s].values()) > v[s]:
                action_to_reward = dict()

                # Calculate at a given state, all action values
                for a in self.mdp.actions(s):
                    reward = 0
                    for ss, p in self.mdp.T(s, a):
                        reward += p * (self.mdp.R(s, a, ss) + v[ss])

                    action_to_reward[a] = reward

                # Find out action with max value
                best_action = max(action_to_reward, key=action_to_reward.get)

                # Greedily select
                for a in self.pi[s]:
                    if a == best_action:
                        self.pi[s][a] = 1
                    else:
                        self.pi[s][a] = 0

        return self.pi

    def check_term(self, v: dict[str, float], next_v: dict[str, float]) -> bool:
        """Return True if the state value has NOT converged.
        Convergence here is defined as follows: 
        For ANY state s, the update delta, abs(v'(s) - v(s)), is within the threshold (self.thresh).

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}
            next_v (dict[str,float]): a state value table (after update)

        Returns:
            bool: True if continue; False if converged
        """
        for s in self.mdp.states():
            if abs(v[s] - next_v[s]) > self.thresh:
                return True

        return False

    def policy_evaluation(self, pi: dict[str, dict[str, float]]):

        for s in self.mdp.states():
            new_val = 0

            for action in pi[s]:
                for ss, t_prob in self.mdp.T(s, action):
                    new_val += pi[s][action] * t_prob * (self.mdp.R(s, action, ss) + self.mdp.gamma * self.v[ss])

            self.v[s] = new_val


class PIAgent(ValueAgent):
    """Policy Iteration Agent class
    """

    def __init__(self, mdp: MDP, conv_thresh: float = 0.000001) -> None:
        """Initialization (Use the functions from the parent class)
        - set up values for member variables
        - init the policy to the random policy

        Args:
            mdp (MDP): An MDP
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """
        super().__init__(mdp, conv_thresh)
        super().init_random_policy()  # initialize its policy function with the random policy
        super().init_random_state()

    def __iter_policy_eval(self, pi: dict[str, dict[str, float]]) -> dict[str, float]:
        """Iterative policy evaluation algorithm. Given a policy pi, evaluate the value of states (v).

        This function should be called in policy_iteration().

        Args:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}

        Returns:
            dict[str,float]: state-value table {state:v-value}
        """

        while True:
            self.v_update_history.append(copy.deepcopy(self.v))
            super().policy_evaluation(pi)
            if not super().check_term(self.v, self.v_update_history[-1]):
                break

        return self.v

    def policy_iteration(self) -> dict[str, dict[str, float]]:
        """Policy iteration algorithm.
        Iterating iter_policy_eval and greedy_policy_improvement, update the policy
        pi until convergence of the state-value function.

        This function is called to run PI.
        e.g.
        mdp = MDP("./mdp1.json")
        dpa = PIAgent(mdp)
        dpa.policy_iteration()

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """

        while True:
            old_policy = copy.deepcopy(self.pi)

            self.__iter_policy_eval(self.pi)
            super().greedy_policy_improvement(self.v)

            if old_policy == self.pi:
                break

        return self.pi


class VIAgent(ValueAgent):
    """Value Iteration Agent class
    """

    def __init__(self, mdp: MDP, conv_thresh: float = 0.000001) -> None:
        """Initialization (Use the functions from the parent class)
        - set up values for member variables
        - init the policy to the random policy

        Args:
            mdp (MDP): An MDP
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """
        super().__init__(mdp, conv_thresh)
        super().init_random_policy()  # initialize its policy function with the random policy
        super().init_random_state()

    def value_iteration(self) -> dict[str, dict[str, float]]:
        """Value iteration algorithm. Compute the optimal v values using the value iteration.
        After that, generate the corresponding optimal policy pi.

        This function is called to run VI. 
        e.g.
        mdp = MDP("./mdp1.json")
        via = VIAgent(mdp)
        via.value_iteration()

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """

        while True:
            old_policy = copy.deepcopy(self.pi)
            self.v_update_history.append(copy.deepcopy(self.v))

            super().policy_evaluation(self.pi)
            super().greedy_policy_improvement(self.v)

            if old_policy == self.pi:
                break

        return self.pi
