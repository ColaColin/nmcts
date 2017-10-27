'''

A class that takes an AbstractLearner and offers functions to play:
- against other NeuralMctsPlayers in large batches
- against humans
- generally a function to find the best move in a given situation, in batches
- used to generate lots of game frames to learn from


This class is where the actual MCTS guided by the neural network happens,
in batches of games which translate to batches of situations to be evaluated at once for the NN

Created on Oct 27, 2017

@author: cclausen
'''

import numpy as np

from MctsTree import TreeNode  # @UnresolvedImport


class NeuralMctsPlayer():
    def __init__(self, mctsExpansions, learner):
        self.mctsExpansions = mctsExpansions # a value of 1 here will make it basically play by the network probabilities in a greedy way #TODO test that
        assert mctsExpansions > 0
        self.learner = learner
        self.cpuct = 1.0 #hmm TODO: investigate the influence of this factor to speed of learning

    def _selectDown(self, node):
        while not node.needsExpand() and not node.state.isTerminal():
            node = node.selectMove(self.cpuct)
        return node

    def _pickMove(self, moveP, state, explore=False):
        ms = []
        ps = []
        psum = 0.0
        for idx, p in enumerate(moveP):
            if state.isMoveLegal(idx):
                ms.append(idx)
                ps.append(p)
                psum += p

        assert len(ms) > 0, "The state should have legal moves or be terminal"
        
        for idx in range(len(ps)):
            ps[idx] /= psum
        
        if explore:
            m = np.random.choice(ms, p = ps)
        else:
            m = ms[np.argmax(ps)]
        return m

    def evaluateByLearner(self, states):
        evalin = [s.state if s != None else None for s in states]
        return self.learner.evaluate(evalin) 

    def batchMcts(self, states):
        """
        runs batched mcts guided by the learner
        yields a result for each state in the batch
        states is expected to be an array of TreeNode(state)
        """
        workspace = states
        for _ in range(self.mctsExpansions):
            workspace = [self._selectDown(s) if s != None else None for s in workspace]
            evalout = self.evaluateByLearner(workspace)
            for idx, ev in enumerate(evalout):
                node = workspace[idx]
                if node == None:
                    continue
                w = ev[1]
                if node.state.isTerminal():
                    w = node.getTerminalResult()
                else:
                    node.expand(ev[0])
                node.backup(w)
                workspace[idx] = states[idx]
    
    def findBestMoves(self, states, noiseMix=0.2):
        """
        searches for the best moves to play in the given states
        this means the move with the most visits in the mcts result greedily is selected
        states is expected to be an array of game states. TreeNodes will be put around them.
        the result is an array of moveIndices
        """
        ts = [TreeNode(s, noiseMix=noiseMix) for s in states]
        self.batchMcts(ts)
        return [self._pickMove(s.getMoveDistribution(), s.state, False) for s in ts]
        
    def playVsHuman(self, state, stateFormatter, commandParser):
        """
        play a game from the given state vs a human using the given command parser function,
        which given a string from input is expected to return a valid move or -1 as a means to signal
        an invalid input.
        stateFormatter is expected to be a function that returns a string given a state
        """
        
    def selfPlayNGames(self, n, batchSize):
        """
        plays n games against itself using more extensive exploration (i.e. pick move probabilistic)
        used to generate games for playing
        """
        
    def playAgainst(self, n, batchSize, others):
        """
        play against other neural mcts players, in batches
        returns the number of wins, losses and draws in that order from the perspective of this player.
        The overall number of players should fit with the game state used. 
        """
    
    
    