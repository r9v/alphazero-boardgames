
class MCTS():
    def __init__(self):
        self.Q = {}  # Q(s,a) - action value
        self.P = {}  # P(s,a) - prior probability
        self.N = {}  # N(s,a) - visit count

    def getPolicy(self, numMCTSSimulations, rootState):
        for i in range(numMCTSSimulations):
            self.search(rootState)

        return policy

    def search(self, state):
       # if state is final return reward
       # select node using UCB

    def UCB(self, s, a):
        return Q[(s, a)] + cput*P[(s, a)]*sqrt()/()
