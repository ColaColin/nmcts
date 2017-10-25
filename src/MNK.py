'''

m,n,k game implementation with the intend of being a simpler test case for the learning algorithm

Created on Oct 21, 2017

@author: cclausen
'''

from NeuralMcts import AbstractState  # @UnresolvedImport # I hate python 'IDEs'
from NeuralMcts import AbstractTorchLearner  # @UnresolvedImport
from NeuralMcts import NeuralMctsTrainer  # @UnresolvedImport

import random

import torch
import torch.nn as nn
import torch.optim as optim
import os
import multiprocessing as mp

'''
The game of MNK
'''
class MNK():
    def __init__(self, m, n, k):
        self.m = m
        self.n = n
        self.k = k
        
        assert m >= k or n >= k
        
        # 0 is an empty field, 1 is the first player, 2 is the second player
        self.board = []
        for _ in range(n):
            self.board.append([-1]*m)
        self.turn = 0
        
        self.winningPlayer = -1
    
    def clone(self):
        c = MNK(self.m, self.n, self.k)
        for y in range(self.n):
            for x in range(self.m):
                c.board[y][x] = self.board[y][x]
        c.turn = self.turn
        c.winningPlayer = self.winningPlayer
        return c
    
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
        self.board[y][x] = (self.turn % 2)
        self.turn += 1
        self._searchWinner(x, y)
    
    def __str__(self):
        mm = ['-', 'X', 'O']
        s = "MNK(%i,%i,%i), " %  (self.m, self.n, self.k)
        if not self.hasEnded():
            s += "On turn: %s\n" % mm[(self.turn % 2)+1]
        elif self.winningPlayer > -1:
            s += "Winner: %s\n" % mm[self.winningPlayer+1]
        else:
            s += "Draw\n"
        for y in range(self.n):
            for x in range(self.m):
                s += mm[self.board[y][x]+1]
            s += "\n"
        return s

def playDemo(m = 3, n = 3, k = 3):
    g = MNK(m,n,k)
    while not g.hasEnded():
        while True:
            rx = random.randint(0, m-1)
            ry = random.randint(0, n-1)
            if g.board[ry][rx] == -1:
                break
        print(g)
        g.place(rx, ry)
    print(g)

class MNKState(AbstractState):
    def __init__(self, mnk):
        self.mnk = mnk
    
    def parseMove(self, move):
        s = move.split("-")
        x = int(s[0])
        y = int(s[1])
        return x, y
    
    def getWinner(self):
        return self.mnk.winningPlayer

    def isMoveLegal(self, move):
        x, y = self.parseMove(move)
        return self.mnk.board[y][x] == -1

    def getPlayerOnTurnIndex(self):
        return self.mnk.turn % 2
    
    def getTurn(self):
        return self.mnk.turn
    
    def isEarlyGame(self):
        return self.mnk.turn < self.mnk.m * self.mnk.n * 0.75
    
    def getPlayerCount(self):
        return 2
    
    def isEqual(self, other):
        assert self.mnk.m == other.mnk.m and self.mnk.n == other.mnk.n and self.mnk.k == other.mnk.k, "Why are these states with different rules even compared?"
        if self.mnk.turn != other.mnk.turn:
            return False
        for idx, myLine in enumerate(self.mnk.board):
            oLine = other.mnk.board[idx]
            for idx in range(self.mnk.m):
                if oLine[idx] != myLine[idx]:
                    return False
        return True
    
    def clone(self):
        return MNKState(self.mnk.clone())
        
    def simulate(self, move):
        x, y = self.parseMove(move)
        self.mnk.place(x, y)
    
    def isTerminal(self):
        return self.mnk.hasEnded()

class MLP(nn.Module):
    def __init__(self, inSize, hiddens, moveSize, winSize):
        super(MLP, self).__init__()
        self.h = nn.Linear(inSize, hiddens)
        self.moveHead = nn.Linear(hiddens, moveSize)
        self.winHead = nn.Linear(hiddens, winSize)
        self.hact = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.hact(self.h(x))
        return self.softmax(self.moveHead(x)), self.softmax(self.winHead(x))

class CNN(nn.Module):
    def __init__(self, inWidth, inHeight, features, hiddens, moveSize, winSize):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, features, 3)
        self.h = nn.Linear(features * (inWidth - 3 + 1) * (inHeight - 3 + 1), hiddens)
        self.moveHead = nn.Linear(hiddens, moveSize)
        self.winHead = nn.Linear(hiddens, winSize)
        self.hact = nn.ReLU()
        self.softmax = nn.Softmax()
        
    
    def forward(self, x):
        x = self.hact(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.hact(self.h(x))
        return self.softmax(self.moveHead(x)), self.softmax(self.winHead(x))

class MNKNetworkLearner(AbstractTorchLearner):
    
    def __init__(self, framesPerIteration, batchSize, epochs, m, n, hiddens, features = -1):
        super(MNKNetworkLearner, self).__init__(framesPerIteration, batchSize, epochs)
        self.m = m
        self.n = n
        self.moveNames = []
        self.hiddens = hiddens
        self.features = features
        for x in range(m):
            for y in range(n):
                self.moveNames.append(str(x) + "-" + str(y))
    
    def clone(self):
        c = MNKNetworkLearner(self.framesPerIteration, self.batchSize, self.epochs, self.m, self.n, self.hiddens, self.features)
        if self.net != None:
            c.initState(None)
            c.net.load_state_dict(self.net.state_dict())
        return c
    
    def getNetInputShape(self):
        if self.features == -1:
            return (self.m * self.n,)
        else:
            return (1, self.m, self.n)
    
    def getPlayerCount(self):
        return 2
    
    def getMoveNames(self):
        return self.moveNames

    def createNetwork(self):
        if self.features == -1:
            return MLP(self.m * self.n, self.hiddens, len(self.moveNames), self.getPlayerCount())
        else:
            return CNN(self.m, self.n, self.features, self.hiddens, len(self.moveNames), self.getPlayerCount())
    
    def createOptimizer(self, net):
        return optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001, nesterov=True)
#         return optim.Adam(net.parameters(), lr=0.001)
    
    def fillNetworkInput(self, state, tensor, batchIndex):
        for x in range(self.m):
            for y in range(self.n):
                b = state.mnk.board[y][x]
                if b != -1:
                    b = state.mapPlayerIndexToTurnRel(b)
                if self.features == -1:
                    tensor[batchIndex,y*self.m+x] = b
                else:
                    tensor[batchIndex,0,x,y] = b
    
    def createEvalNetworkInput(self, state, prevState):  # @UnusedVariable
        if self.features == -1:
            ni = torch.Tensor(1, self.m * self.n)
        else:
            ni = torch.Tensor(1, 1, self.m, self.n)
        self.fillNetworkInput(state, ni, 0)
        return ni
    
m = 5
n = 5
k = 4
features = 128
h = 400

def mnkCreator():
    global m,n,k
    return MNKState(MNK(m,n,k))

def playTrainerVsTrainer(trainerA, trainerB, m, n, k):
    state = MNKState(MNK(m,n,k))
    while not state.isTerminal():
#         print(state.mnk)
        if state.getTurn() % 2 == 0:
            m = trainerA.findCompetitiveMove(state)
        else:
            m = trainerB.findCompetitiveMove(state)
        state.simulate(m)
#     print(state.mnk)
    return state.getWinner()


def playVsAi(trainer, humanIndex=0):
    global m,n,k
    state = MNKState(MNK(m,n,k))
    while not state.isTerminal():
        print(state.mnk)
        if state.getTurn() % 2 == humanIndex:
            tm = trainer.findCompetitiveMove(state)
            print("Trainer would do " + tm)
            m = input("Your move X-Y:")
        else:
            m = trainer.findCompetitiveMove(state)
        state.simulate(m)
    print(state.mnk)

def compareIterations(m, n, k, h, f, basepath, iterA, iterB):
    mlpLearner = MNKNetworkLearner(100000, 100, 50, m,n,h, features=f)
    trainer = NeuralMctsTrainer(mnkCreator, mlpLearner,
                      basepath+str(m)+str(n)+str(k)+"h"+str(h),
                      movesToCheckMin=800, movesToCheckMax=800, moveToCheckIncreasePerIteration=50)
#     mlpLearner.initState(None)
    trainer.loadIteration(iterA)
    
    learnB = MNKNetworkLearner(100000, 100, 50, m,n,h, features=f)
    learnB.initState(os.path.join(trainer.workingdir, "net.iter" + str(iterB)))
    
    print("The champion is iteration %i playing vs the challenger with iteration %i" % (iterA, iterB))
    trainer.isChampionDefeated(mlpLearner, learnB)

# X is iter18
# MNK(3,3,3), On turn: X
# ---
# ---
# ---
# 
# MNK(3,3,3), On turn: O
# ---
# -X-
# ---
# 
# MNK(3,3,3), On turn: X
# --O
# -X-
# ---
# 
# MNK(3,3,3), On turn: O
# --O
# -X-
# -X-
# 
# MNK(3,3,3), On turn: X
# -OO
# -X-
# -X-
# 
# MNK(3,3,3), On turn: O
# -OO
# -XX
# -X-
# 
# MNK(3,3,3), Winner: O
# OOO
# -XX
# -X-

def mkpath(m, n, k, f):
    if f == -1:
        return "/UltraKeks/Dev/git/nmcts/src/models/mnk"+str(m)+str(n)+str(k)+"h"+str(h)
    else:
        return "/UltraKeks/Dev/git/nmcts/src/models/mnk"+str(m)+str(n)+str(k)+"cnn"+str(f)+"h"+str(h)

if __name__ == '__main__':
    mp.set_start_method("spawn")

# VS 6
#     print("0 vs 6 was 77-321")
# Champion won 91 games, challenger won 307 games.
# Champion won 116 games, challenger won 279 games.
# Champion won 132 games, challenger won 256 games.
# Champion won 137 games, challenger won 261 games.



#     for i in range(0, 6):
#         compareIterations(5,5,4,1000,"/UltraKeks/Dev/LiClipse/WorkSpace/Test/src/vindinium/models/mnk", i, 7)
    
    mlpLearnerA = MNKNetworkLearner(150000, 100, 10, m,n,h, features=features)
    trainerA = NeuralMctsTrainer(mnkCreator, mlpLearnerA,
                      mkpath(m, n, k, features),
                      movesToCheckMin=800, movesToCheckMax=1200, moveToCheckIncreasePerIteration=50)
         
#     trainerA.loadIteration(4)
#     mlpLearnerA.initState(None)
#  
#     mlpLearnerB = MNKFlatNetworkLearner(100000, 100, 50, m,n,h)
#     trainerB = NeuralMctsTrainer(mnkCreator, mlpLearnerB,
#                       "/UltraKeks/Dev/LiClipse/WorkSpace/Test/src/vindinium/models/mnk"+str(m)+str(n)+str(k)+"h"+str(h),
#                       movesToCheckMin=400, movesToCheckMax=800, moveToCheckIncreasePerIteration=50)
#        
#     trainerB.loadIteration(18) #TODO last one that is not weird, looks like okay play. What happened in iteration 8?!
#     
#     results = [0,0,0] 
#     for _ in range(100):
#         r = playTrainerVsTrainer(trainerA, trainerB, 3, 3, 3)
#         print(r)
#         results[r] += 1
#     print(results)
        
    
#     playVsAi(trainerA, 0)
#     trainerA.demoPlayGames()
    trainerA.iterateLearning(100, 5000)
