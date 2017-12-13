'''
Created on Dec 11, 2017

@author: cclausen
'''

from nmcts.AbstractTorchLearner import AbstractTorchLearner
from nmcts.NeuralMctsTrainer import NeuralMctsTrainer
from nmcts.NeuralMctsPlayer import NeuralMctsPlayer
from nim.NimGame import NimState, Nim

import torch.nn as nn
import torch.optim as optim

import os

import multiprocessing as mp

import numpy as np

class MLP(nn.Module):
    def __init__(self, inSize, hiddens, moveSize, winSize):
        super(MLP, self).__init__()
        self.h1 = nn.Linear(inSize, hiddens)
        self.h2 = nn.Linear(hiddens, hiddens)
        self.h3 = nn.Linear(hiddens, hiddens)
        self.h4 = nn.Linear(hiddens, hiddens)
        self.h5 = nn.Linear(hiddens, hiddens)
        self.h6 = nn.Linear(hiddens, hiddens)
        self.moveHead = nn.Linear(hiddens, moveSize)
        self.winHead = nn.Linear(hiddens, winSize)
        self.hact = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.hact(self.h1(x))
        x = self.hact(self.h2(x))
        x = self.hact(self.h3(x))
        x = self.hact(self.h4(x))
        x = self.hact(self.h5(x))
        x = self.hact(self.h6(x))
        return self.softmax(self.moveHead(x)), self.softmax(self.winHead(x))
    
def countParams(n):
    r = 0
    for p in n.parameters():
        a = 1
        for s in p.size():
            a *= s
        r += a
    return r

class NimNetworkLearner(AbstractTorchLearner):
    def __init__(self, framesPerIteration, batchSize, epochs, lr_schedule, hiddens, heaps):
        super(NimNetworkLearner, self).__init__(framesPerIteration, batchSize, epochs, lr_schedule)
        self.hiddens = hiddens
        self.heaps = heaps
        self.maxHeap = int(np.max(self.heaps))
        self.initState(None)
    
    def clone(self):
        c = NimNetworkLearner(self.maxFramesLearntPerIteration, self.batchSize, self.epochs, self.lr_schedule, self.hiddens, self.heaps)
        if self.net != None:
            c.initState(None)
            c.net.load_state_dict(self.net.state_dict())
        return c
    
    def getNetInputShape(self):
        return (int(np.sum(self.heaps)), )
        
    def getPlayerCount(self):
        return 2
    
    def getMoveCount(self):
        return self.maxHeap * len(self.heaps)
    
    def createNetwork(self):
        net = MLP(self.getNetInputShape()[0], self.hiddens, self.getMoveCount(), self.getPlayerCount())
        print("Created a network with %i parameters" % countParams(net))
        return net
    
    def createOptimizer(self, net):
        return optim.Adam(net.parameters())
    
    def fillNetworkInput(self, state, tensor, batchIndex):
        acc = 0
        for h in range(len(state.nim.fullHeaps)):
            tensor[batchIndex,acc : (acc + state.nim.fullHeaps[h])] = 0.0
            for i in range(state.nim.heaps[h]):
                tensor[batchIndex, acc + i] = 1.0
            acc += state.nim.fullHeaps[h]

def positionWins(state):
    heaps = state.nim.heaps
    x = 0
    for n in heaps:
        x = x ^ n
    return x > 0
    
def suggestOptimal(state):
    doMove = 0
    
    if not positionWins(state):
        return state.getLegalMoves()[0]
    
    for m in state.getLegalMoves():
        c = state.clone()
        c.simulate(m)
        doMove = m
        if not positionWins(c):
            break
    
    return doMove
    
def stateFormat(state):
    s = str(state.nim)
    if not state.isTerminal():
        smove = state.decodeMoveKey(suggestOptimal(state))
        h = smove[0] + 1
        n = smove[1] + 1
        opt = positionWins(state)
#         s += "\nSuggested move is: " + str(h) + "-" + str(n)
#         s += "\nSituation wins: "+ str(opt)
        s += "\n"
    return s
    
def mkParseCommand(heaps):
    def p(cmd):
        try:
            ms = cmd.split("-")
            h = int(ms[0]) - 1
            n = int(ms[1]) - 1
            return NimState(Nim(heaps)).encodeMoveKey(h, n)
        except:
            return -1
    return p

def mkpath(heaps, hiddens):
    base = "/UltraKeks/Dev/git/nmcts/src/models/nim#"
    return base + str(hiddens)

def verifyOptimalPlay(nplayer, runs, heaps):
    oks = 0
    
    for playerIndex in range(2):
        for r in range(runs):
            state = NimState(Nim(fullHeaps=heaps))
            
            while not state.isTerminal():
                isPlayer = playerIndex == state.getPlayerOnTurnIndex()
                if isPlayer:
                    m = nplayer.findBestMoves([state], noiseMix = 0)
                else:
                    m = suggestOptimal(state)
                state.simulate(m)
            
            if playerIndex == state.getWinner():
                oks += 1
    
    print(oks, runs)
    if oks == runs:
        print("Player played optimal!")
    else:
        print("Player did not play optimal, won only %f" % (oks / runs))

if __name__ == "__main__":
    mp.set_start_method("spawn")
    
    maxIter = 2000
    framesPerIter = 100000
    
    heaps = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    hiddens = 250
    
    lrs = [0.001] * 11 + [0.0001] * maxIter
    epochs = 11
    epochRuns = 2
    bsize = 100
    
    mctsExpansions = 500
    
    cgames = 72
    threads = 4
    
    path = mkpath(heaps, hiddens)
    assert os.path.exists(path), path + " does not exit!"
    
    learner = NimNetworkLearner(framesPerIter, bsize, epochs, lrs, hiddens, heaps)
    player = NeuralMctsPlayer(NimState(Nim(fullHeaps=heaps)), mctsExpansions, learner)
    trainer = NeuralMctsTrainer(player, epochRuns, path, championGames=cgames, batchSize=bsize, threads=threads)
    
#     trainer.iterateLearning(maxIter, 42, startAtIteration=13)
    
    trainer.loadForIteration(14)
    
    trainer.bestPlayer.playVsHuman(NimState(Nim(fullHeaps=heaps)), 0, [], stateFormat, mkParseCommand(heaps))
    
#     verifyOptimalPlay(player, 50, heaps)
    
