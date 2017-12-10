'''
Created on Nov 23, 2017

@author: cclausen
'''

from nmcts.AbstractTorchLearner import AbstractTorchLearner
from nmcts.NeuralMctsTrainer import NeuralMctsTrainer
from nmcts.NeuralMctsPlayer import NeuralMctsPlayer
from connect6.Connect6Game import Connect6, Connect6State

import torch.nn as nn
import torch.optim as optim

import os

import multiprocessing as mp
from matplotlib.font_manager import path

def countParams(n):
    r = 0
    for p in n.parameters():
        a = 1
        for s in p.size():
            a *= s
        r += a
    return r


class ResBlock(nn.Module):
    def __init__(self, features):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(features)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        
        out = self.act(out)
        
        return out

class ResCNN(nn.Module):
    def __init__(self, inWidth, inHeight, baseFeatures, features, blocks, moveSize, winSize):
        super(ResCNN, self).__init__()
        
        assert (inWidth % 2) == (inHeight % 2)
        
        baseKernelSize = 5
        
        self.baseConv = nn.Conv2d(1, baseFeatures, baseKernelSize)
        self.baseBn = nn.BatchNorm2d(baseFeatures)
        self.act = nn.ReLU(inplace=True)
        
        self.matchConv = nn.Conv2d(baseFeatures, features, 1)
        
        blockList = []
        for _ in range(blocks):
            blockList.append(ResBlock(features))
        self.resBlocks = nn.Sequential(*blockList)

        hiddens = features * (inWidth - (baseKernelSize - 1)) * (inHeight - (baseKernelSize - 1))
        self.moveHead = nn.Linear(hiddens, moveSize)
        self.winHead = nn.Linear(hiddens, winSize)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.act(self.baseBn(self.baseConv(x)))
        
        x = self.matchConv(x)
        
        x = self.resBlocks(x)
        
        x = x.view(x.size(0), -1)
        
        moveP = self.softmax(self.moveHead(x))
        winP = self.softmax(self.winHead(x))
        return moveP, winP
    
class C6NetworkLearner(AbstractTorchLearner):
    
    def __init__(self, framesPerIteration, batchSize, epochs, lr_schedule, bfeatures,features, rblocks, m = 19, n = 19):
        super(C6NetworkLearner, self).__init__(framesPerIteration, batchSize, epochs, lr_schedule)
        self.m = m
        self.n = n
        self.features = features
        self.bfeatures = bfeatures
        self.rblocks = rblocks
        self.initState(None)
    
    def clone(self):
        c = C6NetworkLearner(self.maxFramesLearntPerIteration, self.batchSize, 
                              self.epochs, self.lr_schedule, self.bfeatures, self.features, self.rblocks, m=self.m, n=self.n)
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
        net = ResCNN(self.m, self.n, self.bfeatures, self.features, self.rblocks, self.m * self.n, self.getPlayerCount())
        print("Created a network with %i parameters" % countParams(net))
        return net
    
    def createOptimizer(self, net):
        # lr is set before first use elsewhere
#         return optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
        return optim.Adam(net.parameters(), lr=0.001)
    
    def fillNetworkInput(self, state, tensor, batchIndex):
        for y in range(self.n):
            bline = state.c6.board[y]
            for x in range(self.m):
                b = bline[x]
                if b != -1:
                    b = state.mapPlayerIndexToTurnRel(b)
                tensor[batchIndex,0,x,y] = b

def stateFormat(state):
    return str(state.c6)

def mkParseCommand(m,n):
    def p(cmd):
        try:
            ms = cmd.split("-")
            x = int(ms[0]) - 1
            y = int(ms[1]) - 1
            return Connect6State(Connect6(m=m,n=n)).getMoveKey(x,y)
        except:
            return -1
    return p
    
def mkpath(m,n,bf, f, b):
    return "/UltraKeks/Dev/git/nmcts/src/models/c6#"+str(m)+"#"+str(n)+"#"+str(bf)+"#"+str(f)+"#"+str(b)

if __name__ == '__main__':
    mp.set_start_method("spawn")
    
    maxIter = 2000
    framesPerIter = 94570
    gamesPerIter = 42
    
    m = 19
    n = 19
    
    bf = 128
    f = 64
    blocks = 3
    
    lrs = [0.001] * maxIter
    epochs = 16
    epochRuns = 2
    bsize = 70
    mctsExpansions = 1001
    print("Using %i nodes per search tree" % mctsExpansions)
    cgames = 72
    threads = 4
    path = mkpath(m,n,bf,f, blocks)
    assert os.path.exists(path), path + " does not exit!"
    learner = C6NetworkLearner(framesPerIter, bsize, epochs,lrs,bf, f, blocks,m=m, n=n)
    player = NeuralMctsPlayer(Connect6State(Connect6(m=m, n=n)), mctsExpansions, learner)
    trainer = NeuralMctsTrainer(player, epochRuns, path,
                                championGames=cgames, batchSize=bsize, threads=threads)
    trainer.iterateLearning(maxIter, gamesPerIter, startAtIteration=11)
    
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
#     learner0 = C6NetworkLearner(framesPerIter, bsize, epochs, lrs,bf, f, blocks,m=m, n=n)
#     player0 = NeuralMctsPlayer(Connect6State(Connect6(m=m, n=n)), mctsExpansions, learner0)
#     trainer0 = NeuralMctsTrainer(player0, epochRuns, mkpath(m, n, bf, f, blocks),
#                                   championGames=cgames, batchSize = bsize, threads=threads)
#     for base in [10]:
#         trainer0.loadForIteration(base)
#         for i in [5, 8]:
#             trainer.loadForIteration(i)
#             print("Playing with " + str(i))
#             results, _ = trainer.bestPlayer.playAgainst(40, 40, [trainer0.bestPlayer])
#             print("Results %i vs %i are " % (i, base), results)
#             resultsInv, _ = trainer0.bestPlayer.playAgainst(40, 40, [trainer.bestPlayer])
#             print("Results %i vs %i are " % (base, i), resultsInv)
#             iWins = results[0] + resultsInv[1]
#             oWins = results[1] + resultsInv[0]
#             print("Overall win rate of %i vs %i is %f" % (base, i, oWins / float(iWins + oWins)))


##    play vs human
#     trainer.loadForIteration(10)
#     trainer.bestPlayer.playVsHuman(Connect6State(Connect6(m=m, n=n)), 1, [], stateFormat, mkParseCommand(m,n))




