'''
Created on Dec 11, 2017

@author: cclausen
'''

import numpy as np
from nmcts.AbstractState import AbstractState

class Nim():
    def __init__(self, fullHeaps=[1,3,5,7]):
        self.heaps = list(fullHeaps)
        self.fullHeaps = fullHeaps
        self.maxLine = np.max(self.fullHeaps)
        self.turn = 0
        self.winningPlayer = -1
    
    def clone(self):
        c = Nim(fullHeaps = list(self.fullHeaps))
        c.heaps = list(self.heaps)
        c.winningPlayer = self.winningPlayer
        c.turn = self.turn
        return c
    
    def getPlayerIndexOnTurn(self):
        return self.turn % 2
    
    def hasEnded(self):
        return self.winningPlayer != -1
    
    # before calling this one needs to make sure the move is valid!
    def take(self, h, n):
        self.heaps[h] -= (n+1)
        if np.sum(self.heaps) == 0:
            self.winningPlayer = self.getPlayerIndexOnTurn()
        else:
            self.turn += 1
    
    def __str__(self):
        s = "\nNim" + str(self.fullHeaps)+ " Turn " + str(self.turn) + "\n"
        
        for h in self.heaps:
            s += "\n"
            for _ in range(h):
                s += "| "
            if h == 0:
                s += "Empty Line"
        
        if self.winningPlayer != -1:
            s += "\nPlayer " + str(self.winningPlayer) + " has won!"
        
        return s
    
class NimState(AbstractState):
    def __init__(self, nim):
        super(NimState, self).__init__()
        self.nim = nim
        self.legalMoves = None
    
    def canTeachSomething(self):
        return True
    
    def decodeMoveKey(self, key):
        h = int(key / self.nim.maxLine)
        n = int(key % self.nim.maxLine)
        return h, n
    
    def getPlayerOnTurnIndex(self):
        return self.nim.getPlayerIndexOnTurn()
    
    def getTurn(self):
        return self.nim.turn
    
    def getWinner(self):
        return self.nim.winningPlayer
    
    def isEqual(self, _other):
        assert False
    
    def encodeMoveKey(self, h, n):
        return h * self.nim.maxLine + n
    
    def isMoveLegal(self, move):
        h, n = self.decodeMoveKey(move)
        return len(self.nim.heaps) > h and self.nim.heaps[h] > n
    
    def isEarlyGame(self):
        return False
    
    def getPlayerCount(self):
        return 2
    
    def getMoveCount(self):
        return self.nim.maxLine * len(self.nim.heaps)
    
    def clone(self):
        return NimState(self.nim.clone())
    
    def getNewGame(self):
        return NimState(Nim(fullHeaps=self.nim.fullHeaps))
    
    def getFrameClone(self):
        return self.clone()
    
    def augmentFrame(self, frame):
        return frame
    
    def simulate(self, move):
        super(NimState, self).simulate(move)
        h, n = self.decodeMoveKey(move)
        self.nim.take(h, n)
        assert self.legalMoves == None
        
    def isTerminal(self):
        return self.nim.hasEnded()