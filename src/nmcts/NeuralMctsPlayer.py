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

from nmcts.MctsTree import TreeNode

class NeuralMctsPlayer():
    def __init__(self, stateTemplate, mctsExpansions, learner):
        self.stateTemplate = stateTemplate.clone()
        self.mctsExpansions = mctsExpansions # a value of 1 here will make it basically play by the network probabilities in a greedy way #TODO test that
        self.learner = learner
        self.cpuct = 1.0 #hmm TODO: investigate the influence of this factor on the speed of learning

    def clone(self):
        return NeuralMctsPlayer(self.stateTemplate, self.mctsExpansions, 
                                self.learner.clone())

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

        assert len(ms) > 0, "The state should have legal moves"
        
        for idx in range(len(ps)):
            ps[idx] /= float(psum)
        
        if explore:
            m = np.random.choice(ms, p = ps)
        else:
            m = ms[np.argmax(ps)]
        return m

    def evaluateByLearner(self, states):
        evalin = [s.state if s != None else None for s in states]
        return self.learner.evaluate(evalin) 

    def getBatchMctsResults(self, frames, startIndex):
        nodes = [TreeNode(n[0]) for n in frames]
        self.batchMcts(nodes)
        result = []
        for idx, f in enumerate(nodes):
            result.append(( startIndex + idx, f.getMoveDistribution(), f.getBestValue() ))
        return result

    def batchMcts(self, states):
        """
        runs batched mcts guided by the learner
        yields a result for each state in the batch
        states is expected to be an array of TreeNode(state)
        those TreeNodes will be changed as a result of the call
        """
        assert self.mctsExpansions > 0
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
    
    # todo if one could get the caller to deal with the treenode data it might be possible to not throw away the whole tree that was build, increasing play strength
    def findBestMoves(self, states, noiseMix=0.2):
        """
        searches for the best moves to play in the given states
        this means the move with the most visits in the mcts result greedily is selected
        states is expected to be an array of game states. TreeNodes will be put around them.
        the result is an array of moveIndices
        """
        ts = [TreeNode(s, noiseMix=noiseMix) if s != None else None for s in states]
        self.batchMcts(ts)
        return [self._pickMove(s.getMoveDistribution(), s.state, False) if s != None else None for s in ts]
        
    def playVsHuman(self, state, humanIndex, otherPlayers, stateFormatter, commandParser):
        """
        play a game from the given state vs a human using the given command parser function,
        which given a string from input is expected to return a valid move or -1 as a means to signal
        an invalid input.
        stateFormatter is expected to be a function that returns a string given a state
        """
        allPlayers = [self] + otherPlayers
        allPlayers.insert(humanIndex, None)
        
        while not state.isTerminal():
            print(stateFormatter(state))
            player = allPlayers[state.getTurn() % len(allPlayers)]
            if player != None: #AI player
                m = player.findBestMoves([state])[0]
            else:
                m = -1
                while m == -1:
                    m = commandParser(input("Your turn:"))
                    if m == -1:
                        print("That cannot be parsed, try again.")
            state.simulate(m)
            
        print("Game over")
        print(stateFormatter(state))
        
        
    def selfPlayNGames(self, n, batchSize):
        """
        plays n games against itself using more extensive exploration (i.e. pick move probabilistic if state reports early game)
        used to generate games for playing
        """
        
        gamesLeft = n
        gamesRunning = 0
        frames = []
        
        batch = []
        bframes = []
        for _ in range(batchSize):
            if gamesLeft > gamesRunning:
                batch.append(TreeNode(self.stateTemplate.clone()))
                gamesRunning += 1
            else:
                batch.append(None)
            bframes.append([])
        
        while gamesLeft > 0:
            self.batchMcts(batch)
            
            for idx in range(batchSize):
                b = batch[idx]
                if b == None:
                    continue
                md = b.getMoveDistribution()
                if b.state.getTurn() > 0:
                    bframes[idx].append([b.state.clone(), md, b.getBestValue()])
                mv = self._pickMove(md, b.state, b.state.isEarlyGame())
                b = b.getChildForMove(mv)
                
                if b.state.isTerminal():
                    for f in bframes[idx]:
                        frames.append(f + [b.getTerminalResult()])
                    bframes[idx] = []
                    gamesLeft -= 1
                    gamesRunning -= 1
                    if gamesRunning < gamesLeft:
                        batch[idx] = TreeNode(self.stateTemplate.clone())
                        gamesRunning += 1
                    else:
                        batch[idx] = None
                else:
                    batch[idx] = b
                    
                if gamesLeft <= 0:
                    break
                
        return frames
        
        
    def playAgainst(self, n, batchSize, others, collectFrames=False):
        """
        play against other neural mcts players, in batches.
        Since multiple players are used this requires more of a lock-step kind of approach, which makes
        it less efficient than self play!
        returns a pair of:
            the number of wins and draws ordered as [self] + others with the last position representing draws
            a list of lists with the frames of all games, if collectFrames = True
        The overall number of players should fit with the game used.
        No exploration is done here.
        
        !!!Remember that if the game has a first move advantage than one call of this method is probably not enough to fairly compare two players!!!
        """
        
        assert n % batchSize == 0

        batchCount = int(n / batchSize)
        
        results = [0] * (2+len(others)) # the last index stands for draws, which are indexed with -1
        
        allPlayers = [self] + others
        
        gameFrames = []
        if collectFrames:
            for _ in range(n):
                gameFrames.append([])
        
        for bi in range(batchCount):
            
            gamesActive = 0
            batch = []
            
            for _ in range(batchSize):
                batch.append(TreeNode(self.stateTemplate.clone()))
                gamesActive += 1
            
            turn = 0
            
            while gamesActive > 0:
                pIndex = turn % len(allPlayers)
                player = allPlayers[pIndex]
                player.batchMcts(batch)
                turn += 1
                
                for idx in range(batchSize):
                    b = batch[idx]
                    if b == None:
                        continue
                    md = b.getMoveDistribution()
                    
                    gameIndex = batchSize * bi + idx
                    
                    if collectFrames:
                        gameFrames[gameIndex].append([b.state.clone(), md, b.getBestValue()])
                    
                    mv = self._pickMove(md, b.state, False)
                    b = b.getChildForMove(mv)
                    
                    if b.state.isTerminal():
                        
                        if collectFrames:
                            for f in gameFrames[gameIndex]:
                                f.append(b.getTerminalResult())
                                
                        gamesActive -= 1
                        results[b.state.getWinner()] += 1
                        batch[idx] = None
                    else:
                        b.cutTree()
                        batch[idx] = b
        
        return results, gameFrames
        
    