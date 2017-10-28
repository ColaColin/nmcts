'''
Created on Oct 28, 2017

@author: cclausen
'''


from nmcts.AbstractTorchLearner import AbstractTorchLearner
from nmcts.NeuralMctsTrainer import NeuralMctsTrainer
from nmcts.NeuralMctsPlayer import NeuralMctsPlayer
from mnk.MNKGame import MNK, MNKState

import torch.nn as nn
import torch.optim as optim

import multiprocessing as mp

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
        self.moveKeys = MNKState(MNK(m,n,3)).moveKeys
        self.hiddens = hiddens
        self.features = features
        self.initState(None)
    
    def clone(self):
        c = MNKNetworkLearner(self.maxFramesLearntPerIteration, self.batchSize, 
                              self.epochs, self.m, self.n, self.hiddens, self.features)
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
        return len(self.moveKeys)
    
    def createNetwork(self):
        if self.features == -1:
            return MLP(self.m * self.n, self.hiddens, len(self.moveKeys), self.getPlayerCount())
        else:
            return CNN(self.m, self.n, self.features, self.hiddens, len(self.moveKeys), self.getPlayerCount())
    
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
    
def stateFormat(state):
    return str(state.mnk)

def mkParseCommand(m, n, k):
    def p(cmd):
        ms = cmd.split("-")
        x = int(ms[0])
        y = int(ms[1])
        # hmmmmmm
        return MNKState(MNK(m,n,k)).getMoveKey(x,y)
    return p
    
def mkpath(m, n, k, h, f):
    if f == -1:
        return "/UltraKeks/Dev/git/nmcts/src/models/mnk"+str(m)+str(n)+str(k)+"h"+str(h)
    else:
        return "/UltraKeks/Dev/git/nmcts/src/models/mnk"+str(m)+str(n)+str(k)+"cnn"+str(f)+"h"+str(h)
    
if __name__ == '__main__':
    mp.set_start_method("spawn")
    
    m = 5
    n = 5
    k = 4
    f = 128
    h = 400
    
    epochs = 10
    epochRuns = 2
    bsize = 200
    mctsExpansions = 200
    cgames = 500
    threads = 5
    
    learner = MNKNetworkLearner(100000, bsize, epochs, m,n,h,features=f)
    player = NeuralMctsPlayer(MNKState(MNK(m,n,k)), mctsExpansions, learner)
    trainer = NeuralMctsTrainer(player, epochRuns, mkpath(m,n,k,h,f), championGames=cgames, batchSize=bsize, threads=threads)
    
#     player.playVsHuman(MNKState(MNK(m,n,k)), 0, [], stateFormat, mkParseCommand(m, n, k))
    
    trainer.iterateLearning(5000, 20, startAtIteration=0)
