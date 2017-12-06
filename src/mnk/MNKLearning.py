'''
Created on Oct 28, 2017

@author: cclausen
'''


from nmcts.AbstractTorchLearner import AbstractTorchLearner  # @UnresolvedImport
from nmcts.NeuralMctsTrainer import NeuralMctsTrainer  # @UnresolvedImport
from nmcts.NeuralMctsPlayer import NeuralMctsPlayer  # @UnresolvedImport
from mnk.MNKGame import MNK, MNKState  # @UnresolvedImport

import torch.nn as nn
import torch.optim as optim

import multiprocessing as mp

import numpy as np

import os

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
    
    def __init__(self, framesPerIteration, batchSize, epochs, m, n, hiddens, lr_schedule, features = -1):
        super(MNKNetworkLearner, self).__init__(framesPerIteration, batchSize, epochs, lr_schedule)
        self.m = m
        self.n = n
        self.hiddens = hiddens
        self.features = features
        self.initState(None)
    
    def clone(self):
        c = MNKNetworkLearner(self.maxFramesLearntPerIteration, self.batchSize, 
                              self.epochs, self.m, self.n, self.hiddens, self.lr_schedule, self.features)
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
    
    def getMoveCount(self):
        return self.m * self.n
    
    def createNetwork(self):
        if self.features == -1:
            return MLP(self.m * self.n, self.hiddens, self.getMoveCount(), self.getPlayerCount())
        else:
            return CNN(self.m, self.n, self.features, self.hiddens, self.getMoveCount(), self.getPlayerCount())
    
    def createOptimizer(self, net):
        # lr is set before first use elsewhere
        return optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
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
    
def stateFormat(state):
    return str(state.mnk)

def mkParseCommand(m, n, k):
    def p(cmd):
        try:
            ms = cmd.split("-")
            x = int(ms[0]) - 1
            y = int(ms[1]) - 1
            return MNKState(MNK(m,n,k)).getMoveKey(x,y)
        except:
            return -1
    return p
    
def mkpath(m, n, k, h, f):
    if f == -1:
        return "/UltraKeks/Dev/git/nmcts/src/models/mnk"+str(m)+str(n)+str(k)+"h"+str(h)
    else:
        return "/UltraKeks/Dev/git/nmcts/src/models/mnk"+str(m)+str(n)+str(k)+"cnn"+str(f)+"h"+str(h)
    
    
if __name__ == '__main__':
    mp.set_start_method("spawn")
    
    maxIter = 2000
    framesPerIter = 60000
    gamesPerIter = 20
    
    m = 9
    n = 9
    k = 4
    f = 64
    h = 250
    
    lrs = [0.1] * 3 + [0.05] * 10 + [0.005] * 17 + [0.001] * maxIter
    epochs = 10
    epochRuns = 5
    bsize = 200
    mctsExpansions = 542
    print("Using %i nodes per search tree" % mctsExpansions)
    cgames = 100
    threads = 5
    path = mkpath(m,n,k,h,f)
    assert os.path.exists(path), path + " does not exit!"
    learner = MNKNetworkLearner(framesPerIter, bsize, epochs, m,n,h,lrs,features=f)
    player = NeuralMctsPlayer(MNKState(MNK(m,n,k)), mctsExpansions, learner)
    trainer = NeuralMctsTrainer(player, epochRuns, path,
                                championGames=cgames, batchSize=bsize, threads=threads)

    trainer.iterateLearning(maxIter, gamesPerIter, startAtIteration=20)

#     frames = player.selfPlayGamesAsTree(10000, 100)
#     sss = 0
#     for frame in frames:
#         sss += np.sum(frame[3])
# #         print(frame[0].mnk,
# #               frame[1])
# #         print(frame[2:])
# #         print("___")
#         
#     print(sss / len(frames))



# compare different iterations
#     learner0 = MNKNetworkLearner(framesPerIter, bsize, epochs, m, n, h, lrs, features=f)
#     player0 = NeuralMctsPlayer(MNKState(MNK(m,n,k)), mctsExpansions, learner0)
#     trainer0 = NeuralMctsTrainer(player0, epochRuns, mkpath(m, n, k, h, f),
#                                   championGames=cgames, batchSize = bsize, threads=threads)
#     for base in [1, 10, 20]:
#         trainer0.loadForIteration(base)
#         for i in [5,12,13]:
#             trainer.loadForIteration(i)
#             print("Playing with " + str(i))
#             results, _ = trainer.bestPlayer.playAgainst(40, 40, [trainer0.bestPlayer])
#             resultsInv, _ = trainer0.bestPlayer.playAgainst(40, 40, [trainer.bestPlayer])
#             print("Results %i vs %i are " % (i, base), results)
#             print("Results %i vs %i are " % (base, i), resultsInv)
#             iWins = results[0] + resultsInv[1]
#             oWins = results[1] + resultsInv[0]
#             print("Overall win rate of %i vs %i is %f" % (i, base, iWins / float(iWins + oWins)))


##    play vs human
#     trainer.loadForIteration(20)
#     trainer.bestPlayer.playVsHuman(MNKState(MNK(m,n,k)), 1, [], stateFormat, mkParseCommand(m, n, k))




