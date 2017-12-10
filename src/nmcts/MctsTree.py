'''
Created on Oct 27, 2017

@author: cclausen
'''

import numpy as np

from nmcts.AbstractState import AbstractState

import random

class TreeEdge():
    def __init__(self, priorP, parentNode):
        self.visitCount = 0
        self.totalValue = 0
        # TODO have a look at modeling this as a distribution instead of a mean.
        # see arXiv 1707.06887 as detailed inspiration. How to apply that to MCTS?
        self.meanValue = 0.5 #TODO should the default not be 0.5 instead of zero? The values go from 0 to 1 after all. 0 is pretty pessimistic
        self.priorP = priorP
        self.parentNode = parentNode
        self.childNode = None
        
class TreeNode():
    # noiseMix of 0.2 has shown better play-strength than no noise at all in mnk 5,5,4 and is therefore assumed to be a good default to use. #TODO this seems questionable?!
    def __init__(self, state, parentEdge=None, noiseMix = 0.142): 
        assert isinstance(state, AbstractState)
        self.state = state
        mc = state.getMoveCount()
        self.edges = [None] * mc
        self.parent = parentEdge
        self.terminalResult = None
        self.noiseMix = noiseMix
        self.dconst = [0.03] * mc
        self.movePMap = None
        self.isExpanded = False
        self.allVisits = 0
        
        self.hasHighs = False
        self.highs = None
        self.lowS = 0
        self.lowQ = 0
        
    def getBestValue(self):
        bv = 0
        for e in self.edges:
            if e != None and e.meanValue > bv:
                bv = e.meanValue
        return bv
        
    def cutTree(self):
        """
        deletes all children, reducing the tree to the root
        resets all counters
        meant to be used when different solvers are used in an alternating fashion on the same tree.
        maybe instead a completely different tree should be used for each solver. But meh.
        Training does reuse the trees, test play doesn't. Better than nothing...
        """
        self.edges = [None] * self.state.getMoveCount()
        self.parent = None
        self.terminalResult = None
        self.movePMap = None
        self.isExpanded = False
        self.allVisits = 0
        
        self.hasHighs = False
        self.highs = None
        self.lowS = 0
        self.lowQ = 0
        
    def countTreeSize(self):
        c = 1
        for e in self.edges:
            if e != None and e.childNode != None:
                c += e.childNode.countTreeSize()
        return c
    
    def executeMove(self, move):
        assert self.edges[move] != None
        newState = self.state.clone()
        newState.simulate(move)
        return TreeNode(newState, parentEdge=self.edges[move], noiseMix = self.noiseMix)
    
    def getChildForMove(self, move):
        assert self.isExpanded
        
        if self.edges[move] == None:
            self.edges[move] = TreeEdge(self.movePMap[move], self)
        
        child = self.edges[move].childNode
        
        if child == None:
            self.edges[move].childNode = self.executeMove(move)
            child = self.edges[move].childNode 
        
        child.parent = None
        return child
    
    def getMoveDistribution(self):
        sumv = float(self.allVisits)
        
        r = [0] * len(self.edges)
        for m in range(len(r)):
            e = self.edges[m]
            if e != None:
                r[m] = e.visitCount / sumv
        
        return r
    
    def getVisitsFactor(self):
        # .0001 means that in the case of a new node with zero visits it will chose whatever has the best P
        # instead of just the move with index 0
        # but there is little effect in other cases
        return self.allVisits ** 0.5 + 0.0001
    
    def groupCurrentMoves(self, cpuct):
        lMoves = self.state.getLegalMoves()
        numLegalMoves = len(lMoves)
        
        moves = []
        
        for idx in range(numLegalMoves):
            move = lMoves[idx]
            
            e = self.edges[move]
            
            if e != None:
                q = e.meanValue
                p = e.priorP
                vc = e.visitCount
            else:
                q = 0.5
                p = self.movePMap[move]
                vc = 0.0
                
            s = cpuct * p / (1.0+vc)
            
            moves.append((move, q, s))
        
        f = self.getVisitsFactor()
        
        moves.sort(key=lambda x: x[1] + x[2] * f, reverse=True)
        
        lowFactor = 0.9042
        
        highLen = len(moves) - int(len(moves) * lowFactor)
        
        minHighs = 5
        if highLen < minHighs:
            highLen = minHighs
        
        lows = moves[highLen:]
        
        self.highs = list(map(lambda x: x[0], moves[:highLen]))
        self.lowQ = max(map(lambda x: x[1], lows))
        self.lowS = max(map(lambda x: x[2], lows))
        self.hasHighs = True


    def pickMoveFromMoveKeys(self, moveKeys, cpuct):
        allVisitsSq = self.getVisitsFactor()
        
        numKeys = len(moveKeys)
        assert numKeys > 0
        dirNoise = np.random.dirichlet(self.dconst[:numKeys])
        startIdx = random.randint(0, numKeys-1)
        
        moveName = None
        moveValue = 0
        
        for biasedIdx in range(numKeys):
            idx = (biasedIdx + startIdx) % numKeys
            
            iNoise = dirNoise[idx]
            
            idx = moveKeys[idx]
            
            e = self.edges[idx]
            
            if e != None:
                q = e.meanValue
                p = e.priorP
                vc = e.visitCount
            else:
                q = 0.5
                p = self.movePMap[idx]
                vc = 0
            
            p = (1-self.noiseMix) * p + self.noiseMix * iNoise

            u = cpuct * p * (allVisitsSq / (1.0 + vc))
            value = q + u
            if (moveName == None or value > moveValue) and self.state.isMoveLegal(idx):
                moveName = idx
                moveValue = value
                
        return moveName, moveValue
    
    def selectMove(self, cpuct):
        assert self.isExpanded
        
        if not self.hasHighs:
            self.groupCurrentMoves(cpuct)
             
        moveName, fastMoveValue = self.pickMoveFromMoveKeys(self.highs, cpuct)
        lowersBestValue = self.lowQ + self.lowS * self.getVisitsFactor()
        
        if lowersBestValue >= fastMoveValue:
            self.groupCurrentMoves(cpuct)
            moveName, _ = self.pickMoveFromMoveKeys(self.highs, cpuct)
        
#         moveName, _ = self.pickMoveFromMoveKeys(self.state.getLegalMoves(), cpuct)
        
        # this is the slow code replaced by the high-low split of groupCurrentMoves
        # !!!!!! to verify via this assertion remove the randomness from pickMoveFromMoveKeys!
#         moveNameSlow, _ = self.pickMoveFromMoveKeys(self.state.getLegalMoves(), cpuct)
#         assert moveName == moveNameSlow
        
        if self.edges[moveName] == None:
            self.edges[moveName] = TreeEdge(self.movePMap[moveName], self)

        selectedEdge = self.edges[moveName]
        if selectedEdge.childNode == None:
            selectedEdge.childNode = self.executeMove(moveName)
        
        return selectedEdge.childNode
    
    def needsExpand(self):
        return not self.isExpanded

    def backup(self, vs):
        if self.parent != None:
            self.parent.visitCount += 1
            self.parent.parentNode.allVisits += 1
            self.parent.totalValue += vs[self.parent.parentNode.state.getPlayerOnTurnIndex()]
            self.parent.meanValue = float(self.parent.totalValue) / self.parent.visitCount
            self.parent.parentNode.backup(vs)

    def getTerminalResult(self):
        assert self.state.isTerminal()
        if self.terminalResult == None:
            r = [0] * self.state.getPlayerCount()
            winner = self.state.getWinner()
            if winner != -1:
                r[winner] = 1
            else:
                r = [1.0 / self.state.getPlayerCount()] * self.state.getPlayerCount()
            self.terminalResult = r
        return self.terminalResult

    def expand(self, movePMap):
        self.movePMap = movePMap
        self.isExpanded = True

    def getPrevState(self):
        prevState = None
        if self.parent != None:
            prevState = self.parent.parentNode.state
        return prevState
