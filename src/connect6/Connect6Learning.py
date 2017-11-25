'''
Created on Nov 23, 2017

@author: cclausen
'''

from nmcts.AbstractTorchLearner import AbstractTorchLearner  # @UnresolvedImport
from nmcts.NeuralMctsTrainer import NeuralMctsTrainer  # @UnresolvedImport
from nmcts.NeuralMctsPlayer import NeuralMctsPlayer  # @UnresolvedImport
from connect6.Connect6Game import Connect6, Connect6State # @UnresolvedImport

import torch.nn as nn
import torch.optim as optim

import multiprocessing as mp

class CNN(nn.Module):
    def __init__(self, inWidth, inHeight, features, hiddens, moveSize, winSize):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, features, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, features, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, features, 3)
        self.bn3 = nn.BatchNorm2d(16)
        self.h = nn.Linear(features * (inWidth + (-3 + 1) * 3) * (inHeight + (-3 + 1) * 3), hiddens)
        self.moveHead = nn.Linear(hiddens, moveSize)
        self.winHead = nn.Linear(hiddens, winSize)
        self.hact = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.hact(self.bn1(self.conv1(x)))
        x = self.hact(self.bn2(self.conv2(x)))
        x = self.hact(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.hact(self.h(x))
        return self.softmax(self.moveHead(x)), self.softmax(self.winHead(x))
    
class C6NetworkLearner(AbstractTorchLearner):
    
    def __init__(self, framesPerIteration, batchSize, epochs, hiddens, lr_schedule, features, m = 19, n = 19):
        super(C6NetworkLearner, self).__init__(framesPerIteration, batchSize, epochs, lr_schedule)
        self.m = m
        self.n = n
        self.features = features
        self.hiddens = hiddens
        self.initState(None)
    
    def clone(self):
        c = C6NetworkLearner(self.maxFramesLearntPerIteration, self.batchSize, 
                              self.epochs, self.hiddens, self.lr_schedule, self.features,m=self.m, n=self.n)
        if self.net != None:
            c.initState(None)
            c.net.load_state_dict(self.net.state_dict())
        return c
    
    def getNetInputShape(self):
        return (1, self.m, self.n)
    
    def getPlayerCount(self):
        return 2
    
    def getMoveCount(self):
        return self.m * self.n
    
    def createNetwork(self):
        return CNN(self.m, self.n, self.features, self.hiddens, self.m * self.n, self.getPlayerCount())
    
    def createOptimizer(self, net):
        # lr is set before first use elsewhere
        return optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
#         return optim.Adam(net.parameters(), lr=0.001)
    
    def fillNetworkInput(self, state, tensor, batchIndex):
        for x in range(self.m):
            for y in range(self.n):
                b = state.c6.board[y][x]
                if b != -1:
                    b = state.mapPlayerIndexToTurnRel(b)
                tensor[batchIndex,0,x,y] = b
    
def stateFormat(state):
    return str(state.c6)

def mkParseCommand():
    def p(cmd):
        try:
            ms = cmd.split("-")
            x = int(ms[0]) - 1
            y = int(ms[1]) - 1
            return Connect6State(Connect6()).getMoveKey(x,y)
        except:
            return -1
    return p
    
def mkpath(m,n,h, f):
    return "/UltraKeks/Dev/git/nmcts/src/models/c6"+str(m)+"#"+str(n)+"#"+str(f)+"h"+str(h)

if __name__ == '__main__':
    mp.set_start_method("spawn")
    
    maxIter = 2000
    framesPerIter = 200000
    gamesPerIter = 2000
    
    m = 9#19
    n = 9#19
    
    f = 16
    h = 100
    
    lrs = [0.1] * 3 + [0.05] * 10 + [0.005] * 17 + [0.001] * maxIter
    epochs = 5
    epochRuns = 3
    bsize = 200
    mctsExpansions = 800
    print("Using %i nodes per search tree" % mctsExpansions)
    cgames = 80
    threads = 5
    learner = C6NetworkLearner(framesPerIter, bsize, epochs, h,lrs,features=f,m=m, n=n)
    player = NeuralMctsPlayer(Connect6State(Connect6(m=m, n=n)), mctsExpansions, learner)
    trainer = NeuralMctsTrainer(player, epochRuns, mkpath(m,n,h,f),
                                championGames=cgames, batchSize=bsize, threads=threads)
    trainer.iterateLearning(maxIter, gamesPerIter, startAtIteration=0, preloadFrames=None, keepFramesPerc=0.42)
    
    # compare untrained + 5 nodes with untrained + 1500 nodes
#     l0 = C6NetworkLearner(framesPerIter, bsize, epochs, h,lrs,features=f)
#     p0 = NeuralMctsPlayer(Connect6State(Connect6()), 5, l0)
#     t0 = NeuralMctsTrainer(p0, epochRuns, mkpath(h,f), championGames=cgames, batchSize=bsize, threads=threads)
# 
#     l1 = C6NetworkLearner(framesPerIter, bsize, epochs, h,lrs,features=f)
#     p1 = NeuralMctsPlayer(Connect6State(Connect6()), 1500, l1)
#     t1 = NeuralMctsTrainer(p1, epochRuns, mkpath(h,f), championGames=cgames, batchSize=bsize, threads=threads)
# 
#     print(t0.bestPlayer.playAgainst(4, 4, [t1.bestPlayer]))
#     print(t1.bestPlayer.playAgainst(4, 4, [t0.bestPlayer]))

# compare different iterations
#     learner0 = MNKNetworkLearner(framesPerIter, bsize, epochs, m, n, h, lrs, features=f)
#     player0 = NeuralMctsPlayer(MNKState(MNK(m,n,k)), mctsExpansions, learner0)
#     trainer0 = NeuralMctsTrainer(player0, epochRuns, mkpath(m, n, k, h, f),
#                                   championGames=cgames, batchSize = bsize, threads=threads)
#     for base in [5, 10, 15]:
#         trainer0.loadForIteration(base)
#         for i in [19,21,27,30,34]:
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
#     trainer.loadForIteration(34)
#     trainer.bestPlayer.playVsHuman(Connect6State(Connect6()), 1, [], stateFormat, mkParseCommand())




