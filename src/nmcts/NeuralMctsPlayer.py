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

from nmcts.MctsTree import TreeNode  # @UnresolvedImport

import random

import time

import math
from _ast import arg

import os

def getBestNArgs(n, array):
    if len(array) < n:
        return np.arange(0, len(array))
    
    result = [np.argmin(array)] * n
    for idx, pv in enumerate(array):
        rIdx = 0
        isAmongBest = False
        while rIdx < n and pv > array[result[rIdx]]:
            isAmongBest = True
            rIdx += 1
            
        if isAmongBest:
            if rIdx > 1:
                for fp in range(rIdx-1):
                    result[fp] = result[fp+1]
            result[rIdx-1] = idx
    
    return result

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

    def shuffleTogether(self, a, b):
        combined = list(zip(a, b))
        random.shuffle(combined)
        a[:], b[:] = zip(*combined)

    def _pickMoves(self, count, moveP, state, explore=False):
        ms = []
        ps = []
        psum = 0.0
        possibleMoves = 0
        for idx, p in enumerate(moveP):
            if state.isMoveLegal(idx):
                possibleMoves += 1
                eps = 0.0001
                psum += p + eps
                
                ms.append(idx)
                ps.append(p + eps)

        assert len(ms) > 0, "The state should have legal moves"
        
        if possibleMoves <= count:
            return ms

        self.shuffleTogether(ms, ps)

        for idx in range(len(ps)):
            ps[idx] /= float(psum)
        
        if explore:
            m = np.random.choice(ms, count, replace=False, p = ps)
        else:
            if count > 1:
                m = []
                for arg in getBestNArgs(count, ps):
                    m.append(ms[arg])
            else:
                m = [ms[np.argmax(ps)]]
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
        return [self._pickMoves(1, s.getMoveDistribution(), s.state, False)[0] if s != None else None for s in ts]
        
    def playVsHuman(self, state, humanIndex, otherPlayers, stateFormatter, commandParser):
        """
        play a game from the given state vs a human using the given command parser function,
        which given a string from input is expected to return a valid move or -1 as a means to signal
        an invalid input.
        stateFormatter is expected to be a function that returns a string given a state
        """
        allPlayers = [self] + otherPlayers
        if humanIndex != -1:
            allPlayers.insert(humanIndex, None)
        
        while not state.isTerminal():
            print(stateFormatter(state))
            pindex = state.getPlayerOnTurnIndex()
            player = allPlayers[pindex]
            if player != None: #AI player
                m = player.findBestMoves([state])[0]
            else:
                m = -1
                while m == -1:
                    m = commandParser(input("Your turn:"))
                    if m == -1:
                        print("That cannot be parsed, try again.")
            print(pindex+1, m)
            state.simulate(m)
            
        print("Game over")
        print(stateFormatter(state))
        
        # TODO verify this works, then use this on a small test problem to gauge the effect on .. everything
    def selfPlayGamesAsTree(self, collectFrameCount, batchSize, splitsCount=2): #todo. Should be able to set this from somewhere...
        """
        collect frames from games played out in the form of a tree. Results are backpropagated, yielding better estimates about the value of a state
        """
        
        pid = os.getpid()
        
        runningGames = []
        unfinalizedFrames = []
        finalizedFrames = []
        
        roots = 0
        
        lastLog = 5
        
        # collect frames from game trees
        while len(finalizedFrames) < collectFrameCount:
            if lastLog <= len(finalizedFrames):
                print("[Process#%i] Collected %i / %i, running %i games with %i roots" % (pid, len(finalizedFrames), collectFrameCount, len(runningGames), roots))
                lastLog += 500
            
            currentGameCount = len(runningGames)
            
            if currentGameCount == 0 or (currentGameCount < batchSize and random.random() > 0.875):
#                 print("[Process#%i] New Game!!" % (pid))
                runningGames.append(TreeNode(self.stateTemplate.getNewGame()))
                unfinalizedFrames.append(None)
            
            currentGameCount = len(runningGames)
            
            newRunningGames = []
            newUnfinalFrames = []

            # do mcts searches for all current states            
            batchCount = math.ceil(currentGameCount / batchSize)
            for idx in range(batchCount):
                startIdx = idx * batchSize
                endIdx = (idx+1) * batchSize
                batch = runningGames[startIdx:endIdx]
                self.batchMcts(batch)
            
            eSplitAdd = 0 #max(0, batchSize - currentGameCount)
            
            if batchSize > currentGameCount:
                eSplitAdd = 1
            else:
                eSplitAdd = -1
            
            # find the indices of games to be split in this iteration
            splitIdx = random.randint(0, currentGameCount)
            splitIdxs = []
            for i in range(splitsCount + eSplitAdd):
                offset = 0
                for offset in range(currentGameCount + 1):
                    nextIdx = (splitIdx + i + offset) % currentGameCount
                    if offset >= currentGameCount or (unfinalizedFrames[nextIdx] != None and unfinalizedFrames[nextIdx][6] > 3 and unfinalizedFrames[nextIdx][5] < 2):
                        splitIdxs.append(nextIdx)
                        break

            # iterate all current states
            for gidx in range(currentGameCount):
                g = runningGames[gidx]
                md = g.getMoveDistribution()

                assert np.sum(md) > 0
                
                splitExtra = 0
                for sIdx in splitIdxs:
                    if gidx == sIdx:
                        splitExtra += 1
                
                mvs = self._pickMoves(1 + splitExtra, md, g.state, g.state.isEarlyGame())
                
                parentStateFrame = unfinalizedFrames[gidx]

                ancestor = parentStateFrame
                while ancestor != None:
                    ancestor[5] += len(mvs) - 1
                    ancestor = ancestor[4]
                
                if g.state.getTurn() > 0:
                    isSplitting = len(mvs) > 1
                    turnsSinceSplit = 0
                    
                    if not isSplitting and parentStateFrame != None:
                        turnsSinceSplit = parentStateFrame[6] + 1
                    
                    stateFrame = [g.state.getFrameClone(),
                                  md,
                                  g.getBestValue(),
                                  np.array([0.0] * g.state.getPlayerCount()),
                                  parentStateFrame,
                                  len(mvs),
                                  turnsSinceSplit]
                else:
                    stateFrame = None
                
                for mv in mvs:
                    nextState = g.getChildForMove(mv)
                    
                    if nextState.state.getTurn() == 1:
                        roots += 1
                    
                    if not nextState.state.isTerminal():
                        newRunningGames.append(nextState)
                        newUnfinalFrames.append(stateFrame)
                    else:
                        stateResult = np.array(nextState.getTerminalResult())
                        
                        ancestor = stateFrame
                        depth = 0
                        while ancestor != None:
                            ancestor[3] += stateResult
                            depth += 1
                            if np.sum(ancestor[3]) > ancestor[5] - 0.5:
                                eps = 0.000000001
                                ancestor[3] = (ancestor[3]) / (np.sum(ancestor[3]) + eps)
                                if depth <= nextState.state.getPlayerCount() or ancestor[5] > 1:
                                    finalizedFrames.append(ancestor[:4])

                                if ancestor[4] == None:
                                    roots -= 1
                                    assert roots > -1
                            ancestor = ancestor[4]
                            
            
            runningGames = newRunningGames
            unfinalizedFrames = newUnfinalFrames
        
        return finalizedFrames
        
    def selfPlayNGames(self, n, batchSize, keepFramesPerc = 1.0):
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
                initialGameState = self.stateTemplate.getNewGame()
                batch.append(TreeNode(initialGameState))
                gamesRunning += 1
            else:
                batch.append(None) # TODO why this hacky stuff with NONE anyway?!
            bframes.append([])
        
        while gamesLeft > 0:
#             print("Begin batch")
#             t = time.time()
            self.batchMcts(batch)
#             print("Batch complete in %f" % (time.time() - t))
            
            for idx in range(batchSize):
                b = batch[idx]
                if b == None:
                    continue
                md = b.getMoveDistribution()
                if b.state.getTurn() > 0:
                    bframes[idx].append([b.state.getFrameClone(), md, b.getBestValue()])
                mv = self._pickMoves(1, md, b.state, b.state.isEarlyGame())[0]
                b = b.getChildForMove(mv)
                
                if b.state.isTerminal():
                    for f in bframes[idx]:
                        if keepFramesPerc == 1.0 or random.random() < keepFramesPerc:
                            frames.append(f + [b.getTerminalResult()])
                    bframes[idx] = []
                    gamesLeft -= 1
                    gamesRunning -= 1
                    if gamesRunning < gamesLeft:
                        batch[idx] = TreeNode(self.stateTemplate.getNewGame())
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
                initialGameState = self.stateTemplate.getNewGame()
                batch.append(TreeNode(initialGameState))
                gamesActive += 1
            
            while gamesActive > 0:
                someGame = None
                for b in batch:
                    if b != None:
                        someGame = b
                        break
                # the games are all in the same turn
                pIndex = someGame.state.getPlayerOnTurnIndex()
                for b in batch:
                    if b != None:
                        assert b.state.getPlayerOnTurnIndex() == pIndex
                        
                player = allPlayers[pIndex]
                player.batchMcts(batch)
                
#                 print(pIndex, someGame.state.c6, ["{0:.5f}".format(x) for x in someGame.getMoveDistribution()])
                
                for idx in range(batchSize):
                    b = batch[idx]
                    if b == None:
                        continue
                    md = b.getMoveDistribution()
                    
                    gameIndex = batchSize * bi + idx
                    
                    if collectFrames:
                        gameFrames[gameIndex].append([b.state.getFrameClone(), md, b.getBestValue()])
                    
                    mv = self._pickMoves(1, md, b.state, False)[0]
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
        
    