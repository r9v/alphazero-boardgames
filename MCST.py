
class MCTS():
    def __init__(self):
        self.Q = {}  # Q(s,a) - action value
        self.Pi = []  # Pi(s) - policy
        self.N = {}  # N(s,a) - visit count
        self.visited = []

    def getPolicy(self, numMCTSSimulations, rootState):
        for i in range(numMCTSSimulations):
            self.search(rootState)

        return policy

    def search(self, state):

        # if state is final return reward
        gameOver, reward = game.over(state)
        if gameOver:
            return reward

        # if node not explored get values from nnet
        if state not in self.visited:
            self.visited.append(state)
            # todo mask Pi with only valid moves
            self.Pi[state], v = nnet.predict()
            return v

        # do search on child node selected by using UCB
        bestAction = self.UCB(state)
        nextState = game.nextState(state, bestAction)
        nextStateValue = self.search(nextState)
        Q[(state, bestAction)] = nextStateValue  # do average instead
        N[(state, bestAction)] += 1

    def UCB(self, s):
        return Q[(s, a)] + cput*P[(s, a)]*sqrt()/()
