'''
Created on Nov 23, 2017

@author: cclausen
'''

from nmcts.AbstractState import AbstractState
from nmcts.FieldAugments import initField, augmentFieldAndMovesDistribution

import math

class Connect6():
    def __init__(self, m=19, n = 19):
        self.m = m
        self.n = n
        self.k = 6
        
        assert m >= self.k or n >= self.k
        
        # -1 is an empty field, 0 is white, 1 is black
        self.board = initField(m, n, -1)
        self.turn = 0
        
        self.winningPlayer = -1
    
    def clone(self):
        c = Connect6(m = self.m, n = self.n)
        for y in range(self.n):
            for x in range(self.m):
                c.board[y][x] = self.board[y][x]
        c.turn = self.turn
        c.winningPlayer = self.winningPlayer
        return c
    
    def getPlayerIndexOnTurn(self):
        if self.turn == 0:
            return 1
        else:
            return math.floor((self.turn-1)/2) % 2
    
    def hasEnded(self):
        return self.winningPlayer != -1 or self.turn >= self.m * self.n
    
    def _searchWinner(self, lx, ly):
        """
        return -1 if the game is not over
        else the number of the winning player
        """
        if self.winningPlayer != -1:
            return
        
        p = self.board[ly][lx]
        if p != -1:
            dirs = [(1,0),(0,1),(1,-1),(1,1)]
            for direction in dirs:
                l = 0
                for d in [1,-1]:
                    x = lx
                    y = ly
                    while x > -1 and y > -1 and x < self.m and y < self.n and self.board[y][x] == p:
                        l += 1
                        x += d * direction[0]
                        y += d * direction[1]
                        
                        if l - 1 >= self.k:
                            self.winningPlayer = p
                            return
     
    def place(self, x, y):
        self.board[y][x] = self.getPlayerIndexOnTurn()
        self.turn += 1
        self._searchWinner(x, y)
    
    def __str__(self):
        mm = ['-', 'X', 'O']
        s = "Connect6(%i,%i), " %  (self.m, self.n)
        if not self.hasEnded():
            s += "Turn %i: %s\n" % (self.turn, mm[self.getPlayerIndexOnTurn()+1])
        elif self.winningPlayer > -1:
            s += "Winner: %s\n" % mm[self.winningPlayer+1]
        else:
            s += "Draw\n"
        for y in range(self.n):
            for x in range(self.m):
                s += mm[self.board[y][x]+1]
#                 if (x+1) % 5 == 0:
#                     s += " "
            s += "\n"
#             if (y+1) % 5 == 0:
#                 s += "\n"
        return s
    
class Connect6State(AbstractState):
    def __init__(self, c6):
        super(Connect6State, self).__init__()
        self.c6 = c6
        self.legalMoves = None
        
    def canTeachSomething(self):
        return True
        
    def getWinner(self):
        return self.c6.winningPlayer

    def getMoveLocation(self, key):
        y = int(key / self.c6.m)
        x = int(key % self.c6.m)
        return x, y

    def getMoveKey(self, x, y):
        return y * self.c6.m + x

    def isMoveLegal(self, move):
        x, y = self.getMoveLocation(move)
        return self.c6.board[y][x] == -1

    def getPlayerOnTurnIndex(self):
        return self.c6.getPlayerIndexOnTurn()
    
    def getTurn(self):
        return self.c6.turn
    
    def isEarlyGame(self):
        # TODO: hmmmm how does this affect things actually?!
        c = max(self.c6.m, self.c6.n)
        c += c % 2
        return self.c6.turn < c
    
    def getPlayerCount(self):
        return 2
    
    def getMoveCount(self):
        return self.c6.m * self.c6.n
    
    def isEqual(self, other):
        assert self.c6.m == other.c6.m and self.c6.n == other.c6.n and self.c6.k == other.c6.k, "Why are these states with different rules even compared?"
        if self.c6.turn != other.c6.turn:
            return False
        for idx, myLine in enumerate(self.c6.board):
            oLine = other.c6.board[idx]
            for idx in range(self.c6.m):
                if oLine[idx] != myLine[idx]:
                    return False
        return True
    
    def clone(self):
        return Connect6State(self.c6.clone())
    
    def getNewGame(self):
        return Connect6State(Connect6(self.c6.m, self.c6.n))
    
    def getFrameClone(self):
        return self.clone()
        
    def augmentFrame(self, frame):
        frame = [frame[0].clone(), list(frame[1]), frame[2], list(frame[3])]
        
        fState = frame[0].c6
        fMoves = frame[1]

        augmentFieldAndMovesDistribution(fState.m, fState.n, fState.board, fMoves, 
                                         lambda idx: self.getMoveLocation(idx),
                                         lambda x,y: self.getMoveKey(x, y))

        return frame
        
    def simulate(self, move):
        super(Connect6State, self).simulate(move)
        x, y = self.getMoveLocation(move)
        self.c6.place(x, y)
        assert self.legalMoves == None
    
    def isTerminal(self):
        return self.c6.hasEnded()
    
    def mapPlayerIndexToTurnRel(self, playerIndex):
        onTurnIdx = self.getPlayerOnTurnIndex()
        if onTurnIdx == playerIndex:
            return 0
        else:
            return 1
        
        