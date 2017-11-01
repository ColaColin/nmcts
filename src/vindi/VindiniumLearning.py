'''
Created on Nov 1, 2017

@author: cclausen
'''

from nmcts.AbstractTorchLearner import AbstractTorchLearner 
from nmcts.NeuralMctsTrainer import NeuralMctsTrainer
from nmcts.NeuralMctsPlayer import NeuralMctsPlayer

from vindi.VindiniumGame import VindiniumState

import torch.nn as nn
import torch.optim as optim

import multiprocessing as mp


class CNN(nn.Module):
    def __init__(self,f=1):
        super(CNN, self).__init__()
        # TODO this breaks with an internal CUDNN error
        # something is bad about this
        self.conv1 = nn.Conv2d(7, f*64, 3) # 28 -> 26
        self.bn1 = nn.BatchNorm2d(f*64)
        self.conv2 = nn.Conv2d(f*64, f*64, 3) # 26 -> 24
        self.bn2 = nn.BatchNorm2d(f*64)
        self.conv3 = nn.Conv2d(f*64, f*32, 5) # 24 -> 20
        self.bn3 = nn.BatchNorm2d(f*32)
        self.conv4 = nn.Conv2d(f*32, f*32, 5) # 20 -> 16
        self.bn4 = nn.BatchNorm2d(f*32)
        self.conv5 = nn.Conv2d(f*32, f*16, 5) # 16 -> 12
        self.bn5 = nn.BatchNorm2d(f*16)
        self.conv6 = nn.Conv2d(f*16, f*16, 5) # 12 -> 8
        self.bn6 = nn.BatchNorm2d(f*16)
        self.conv7 = nn.Conv2d(f*16, f*8, 5) # 8 -> 4
        self.bn7 = nn.BatchNorm2d(f*8)
        self.moveHead = nn.Linear(f*8 * 4 * 4, 5)
        self.winHead = nn.Linear(f*8 * 4 * 4, 4)
        self.hact = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.hact(self.bn1(self.conv1(x)))
        x = self.hact(self.bn2(self.conv2(x)))
        x = self.hact(self.bn3(self.conv3(x)))
        x = self.hact(self.bn4(self.conv4(x)))
        x = self.hact(self.bn5(self.conv5(x)))
        x = self.hact(self.bn6(self.conv6(x)))
        x = self.hact(self.bn7(self.conv7(x)))
        x = x.view(x.size(0), -1)
        moveP = self.softmax(self.moveHead(x))
        winP = self.softmax(self.winHead(x))
        return moveP, winP 
    
class VindiniumLearner(AbstractTorchLearner):
    def __init__(self,framesPerIteration, batchSize, epochs, lr_schedule):
        super(VindiniumLearner, self).__init__(framesPerIteration, batchSize, epochs, lr_schedule)
        self.initState(None)
        
    def clone(self):
        c = VindiniumLearner(self.maxFramesLearntPerIteration, self.batchSize,
                             self.epochs, self.lr_schedule)
        if self.net != None:
            c.initState(None)
            c.net.load_state_dict(self.net.state_dict())
        return c

    def getNetInputShape(self):
        return (7, 28, 28)
    
    def getPlayerCount(self):
        return 4
        
    def getMoveCount(self):
        return 5
    
    def createNetwork(self):
        return CNN(f=1)
    
    def createOptimizer(self, net):
        # lr is set before first use elsewhere
        return optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
    
    def fillNetworkInput(self, state, tensor, batchIndex):
        tensor[batchIndex] = state.tensor
        
def mkpath(f, name):
    return "/UltraKeks/Dev/git/nmcts/src/models/vindi"+name+str(f)
    
def stateFormat(state):
    return str(state)

def parseCommand(cmd):
    NORTH = 'North'
    SOUTH = 'South'
    WEST  = 'West'
    EAST  = 'East'
    STAY  = 'Stay'
    return [NORTH, SOUTH, WEST, EAST, STAY].index(cmd)
    
if __name__ == '__main__':
    mp.set_start_method("spawn")
    
    name = "test"
    maxIter = 2000
    gamesPerIter = 20
    turnsPerGame = 20
    framesPerIter = 100000
    f = 1
    
    lrs = [0.1] * 3 + [0.05] * 10 + [0.005] * 17 + [0.001] * maxIter
    epochs = 10
    epochRuns = 2
    
    bsize = 25
    mctsExpansions = 500
    
    print("Using %i nodes per search tree" % mctsExpansions)
    cgames = 20
    threads = 4
    
    learner = VindiniumLearner(framesPerIter, bsize, epochs, lrs)
    player = NeuralMctsPlayer(VindiniumState(maxTurns=turnsPerGame, isTemplate=True), mctsExpansions, learner)
    trainer = NeuralMctsTrainer(player, epochRuns, mkpath(f, name),
                                championGames=cgames, batchSize=bsize,threads=threads)
    
    trainer.iterateLearning(maxIter, gamesPerIter, startAtIteration=0)
    
#     trainer.bestPlayer.playVsHuman(VindiniumState(maxTurns=40), 0, [trainer.bestPlayer,trainer.bestPlayer], stateFormat, parseCommand)
    