class mySmartAgent:
    def __init__(self):
        self.statement = "I am the smarty pantys pants"

    def make_a_move(self, observation):
        # can simply do a greedy
        pass

class myDumbAgent:
    # agent need to know the action space

    def __init__(self, ID=None, action_space=None):
        self.ID = ID
        self.action_space = action_space




    def random_action(self, action_space = None,observation=None):
        # random mov
        #randomly choose from available spot from obseravtion
        print("random")
        return

