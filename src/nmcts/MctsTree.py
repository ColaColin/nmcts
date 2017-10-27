'''
Created on Oct 27, 2017

@author: cclausen
'''

import numpy as np

from AbstractState import AbstractState  # @UnresolvedImport


class TreeEdge():
    def __init__(self, priorP, parentNode):
        self.visitCount = 0
        self.totalValue = 0
        self.meanValue = 0 #TODO should the default not be 0.5 instead of zero? The values go from 0 to 1 after all. 0 is pretty pessimistic
        self.priorP = priorP
        self.parentNode = parentNode
        self.childNode = None
        
class TreeNode():
    # noiseMix of 0.2 has shown better play-strength than no noise at all in mnk 5,5,4 and is therefore assumed to be a good default to use
    def __init__(self, state, parentEdge=None, noiseMix = 0.2):
        assert isinstance(state, AbstractState)
        self.state = state
        self.edges = [None] * state.getMoveCount()
        self.parent = parentEdge
        self.terminalResult = None
        self.noiseMix = noiseMix
        self.dconst = [0.03] * self.state.getMoveCount()
        self.movePMap = None
        self.isExpanded = False
        self.allVisits = 0
        
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
    
    def getWinnerEncoding(self):
        r = [0] * self.state.getPlayerCount()
        winner = self.state.getWinner()
        if winner != -1:
            r[winner] = 1
        else:
            for idx in range(self.state.getPlayerCount()):
                r[idx] = 1.0 / self.state.getPlayerCount()
        return r
    
    def selectMove(self, cpuct):
        moveName = None
        moveValue = 0
        allVisits = self.allVisits
        
        allVisitsSq = allVisits ** 0.5

        dirNoise = np.random.dirichlet(self.dconst)
        
        for idx in range(self.state.getMoveCount()):
            if not self.state.isMoveLegal(idx):
                continue
            
            e = self.edges[idx]
            
            if e != None:
                q = e.meanValue
                p = e.priorP
                vc = e.visitCount
            else:
                q = 0
                p = self.movePMap[idx]
                vc = 0
            
            p = (1-self.noiseMix) * p + self.noiseMix * dirNoise[idx]
            
            # .01 means that in the case of a new node with zero visits it will chose whatever has the best P
            # instead of just the move with index 0
            # but there is little effect in other cases
            u = cpuct * p * ((.01 + allVisitsSq) / (1 + vc))
            value = q + u
            if (moveName == None or value > moveValue):
                moveName = idx
                moveValue = value

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
