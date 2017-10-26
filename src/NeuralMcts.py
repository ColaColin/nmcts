'''
Created on Oct 21, 2017

@author: cclausen
'''

import abc
import numpy as np
import time
import random

import os

import torch
from torch.autograd import Variable

import multiprocessing as mp

class AbstractState(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def getWinner(self):
        """
        return index of the winning player or -1 if a draw
        """
    
    def isMoveLegal(self, move):
        """
        return True if the given move is legal
        """
    
    @abc.abstractmethod
    def getPlayerOnTurnIndex(self):
        """
        return index of player whose turn it is right now
        """
    
    @abc.abstractmethod
    def getTurn(self):
        """
        return the current turn, the first turn is turn 0
        """
        
    def isEarlyGame(self):
        """
        return if moves should be deterministic(False) or probabilistic (True)
        """
    
    @abc.abstractmethod
    def getPlayerCount(self):
        """
        returns the number of players who play this game
        """
    
    @abc.abstractmethod
    def clone(self):
        """
        returns a deep copy of the state
        """
    
    @abc.abstractmethod
    def simulate(self, move):
        """
        Do one step of the simulation given a move for the current player.
        Mutate this object.
        """
    
    @abc.abstractmethod
    def isEqual(self, other):
        """
        returns if this state is equal to the given other state. 
        Used to generate statistics about the diversity of encountered states
        """
    
    @abc.abstractmethod 
    def isTerminal(self):
        """
        return true iff this state is terminal, i.e. additional moves are not allowed
        """
        
    def mapPlayerIndexToTurnRel(self, playerIndex):
        """
        return playerIndex -> playerIndex relative to current turn 
        """
        pc = self.getPlayerCount()
        return (playerIndex - (self.getTurn() % pc)) % pc

class AbstractLearner(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def initState(self, file):
        """
        initialize the learner. file path may be None or may point at a file storing learner state
        """
        
    @abc.abstractmethod
    def saveState(self, file):
        """
        save current state to given file path
        """
    
    @abc.abstractmethod
    def evaluate(self, state, prevState):
        """
        returns a dict moveName -> moveProbability
        and values from the perspective of each player as an array. Values should be winning probabilities from 0 to 1
        for the AbstractState state. prevState is the parent state and may be helpful to optimize things, but it can be ignored
        """
        
    @abc.abstractmethod
    def clone(self):
        """
        returns a copy of the evaluator, to be used as a checkpoint or comparison
        """
        
    @abc.abstractmethod
    def learnFromFrames(self, frames, dbg=False):
        """
        learn from the given frames. Each frame is a tripel of
        state, moveP, winP
        where moveP and winP are the learning targets for state
        """
        
    def getFramesPerIteration(self):
        """
        returns the maximum number of frames learned from per iteration
        """
        
class AbstractTorchLearner(AbstractLearner, metaclass=abc.ABCMeta):
    def __init__(self, framesPerIteration, batchSize, epochs):
        assert framesPerIteration % batchSize == 0

        self.framesPerIteration = framesPerIteration
        self.batchSize = batchSize
        self.epochs = epochs
    
    def getFramesPerIteration(self):
        return self.framesPerIteration
    
    @abc.abstractmethod
    def getNetInputShape(self):
        """
        returns a tuple that describes the input shape of the network, minus the batchdimension
        """
    
    @abc.abstractmethod
    def getPlayerCount(self):
        """
        returns the number of players
        """

    @abc.abstractmethod
    def getMoveNames(self):
        """
        returns an array of all possibles move names
        """
    
    @abc.abstractmethod
    def createNetwork(self):
        """
        return a newly created Torch Network
        """
    
    @abc.abstractmethod
    def createOptimizer(self, net):
        """
        return a torch optimizer to be used for the learning process
        """
    
    @abc.abstractmethod
    def fillNetworkInput(self, state, tensor, batchIndex):
        """
        fill the given tensor with the input that represents state at the given batchIndex.
        The tensor is zero'd out before this is called
        """
        
    @abc.abstractmethod
    def createEvalNetworkInput(self, state, prevState):
        """
        return a tensor that describes state to the network. prevState may be None or
        may contain a previous state to state, which may be helpful to optimize things
        """
    
    def initState(self, file):
        self.networkInput = torch.zeros((self.framesPerIteration,) + self.getNetInputShape()).pin_memory()
        self.moveOutput = torch.zeros(self.framesPerIteration, len(self.getMoveNames())).pin_memory()
        self.winOutput = torch.zeros(self.framesPerIteration, self.getPlayerCount()).pin_memory()
        self.net = self.createNetwork()
        self.opt = self.createOptimizer(self.net)
        
        if file != None:
            self.net.load_state_dict(torch.load(file))
            print("Loaded state from " + file)
        
        self.net.cuda()
    
    def saveState(self, file):
        torch.save(self.net.state_dict(), file)
    
    def evaluate(self, state, prevState):
        netIn = self.createEvalNetworkInput(state, prevState)
        moveP, winP = self.net(Variable(netIn).cuda())

        r = dict()
        for m in self.getMoveNames():
            r[m] = moveP.data[0, self.getMoveNames().index(m)]

        w = []
        
        for pid in range(state.getPlayerCount()):
            w.append(winP.data[0, state.mapPlayerIndexToTurnRel(pid)])
        
        return r, w
    
    def fillTrainingSet(self, frames):
        self.moveOutput.fill_(0)
        self.winOutput.fill_(0)
        self.networkInput.fill_(0)
        
        for fidx, frame in enumerate(frames):
            self.fillNetworkInput(frame[0], self.networkInput, fidx)
            
            for mk in dict.keys(frame[1]):
                self.moveOutput[fidx, self.getMoveNames().index(mk)] = frame[1][mk]
            
            for pid in range(frame[0].getPlayerCount()):
                self.winOutput[fidx, frame[0].mapPlayerIndexToTurnRel(pid)] = frame[2][pid]
    
    def learnFromFrames(self, frames, dbg=False):
        assert(len(frames) <= self.framesPerIteration)
        self.fillTrainingSet(frames)
        
        batchNum = int(len(frames) / self.batchSize)
        
        if dbg:
            print(len(frames), self.batchSize, batchNum)

        print("Running nan asserts....")
        assert torch.sum(self.networkInput.ne(self.networkInput)) == 0
        assert torch.sum(self.moveOutput.ne(self.moveOutput)) == 0
        assert torch.sum(self.winOutput.ne(self.winOutput)) == 0
        print("DONE")
        
        nIn = Variable(self.networkInput).cuda()
        mT = Variable(self.moveOutput).cuda()
        wT = Variable(self.winOutput).cuda()
        
        for e in range(self.epochs):
            mls = []
            wls = []
            
            for bi in range(batchNum):
                self.opt.zero_grad()
                
                x = nIn[bi*self.batchSize : (bi+1) * self.batchSize]
                yM = mT[bi*self.batchSize : (bi+1) * self.batchSize]
                yW = wT[bi*self.batchSize : (bi+1) * self.batchSize] 
                
                if dbg:
                    print(x, yM, yW)
                
                mO, wO = self.net(x)
                
                mLoss = -torch.sum(torch.log(mO) * yM) / self.batchSize
                wLoss = -torch.sum(torch.log(wO) * yW) / self.batchSize
                
                loss = mLoss + wLoss
                loss.backward()
                
                self.opt.step()
                
                mls.append(mLoss.data[0])
                wls.append(wLoss.data[0])
                
            print("Completed Epoch %i with loss %f + %f" % (e, np.mean(mls), np.mean(wls)))
            
        del nIn
        del mT
        del wT

class TreeEdge():
    def __init__(self, priorP, parentNode):
        self.visitCount = 0
        self.totalValue = 0
        self.meanValue = 0
        self.priorP = priorP
        self.parentNode = parentNode
        self.childNode = None
        
class TreeNode():
    def __init__(self, state, parentEdge=None):
        assert isinstance(state, AbstractState)
        
        self.state = state
        self.edges = dict()
        self.moveOptions = []
        self.parent = parentEdge
        
    def countTreeSize(self):
        c = 1
        for e in dict.values(self.edges):
            if e.childNode != None:
                c += e.childNode.countTreeSize()
        return c
    
    def executeMove(self, move):
        newState = self.state.clone()
        newState.simulate(move)
        return TreeNode(newState, parentEdge=self.edges[move])
    
    def getChildForMove(self, move):
        child = self.edges[move].childNode
        
        if child == None:
            self.edges[move].childNode = self.executeMove(move)
            child = self.edges[move].childNode 
        
        child.parent = None
        return child
    
    def selectMove(self, cpuct):
        moveName = None
        moveValue = 0
        allVisits = 0
        
        for e in dict.values(self.edges):
            allVisits += e.visitCount
        allVisitsSq = 1 + allVisits ** 0.5
        
#         foo = []
        
        #if self.parent == None: # and explore: #hmmmmm (also make sure to check below where this value is then actually used!
        dirNoise = np.random.dirichlet([0.03] * len(self.moveOptions))
        
        for idx, m in enumerate(self.moveOptions):
            if not self.state.isMoveLegal(m):
                continue
            
            e = self.edges[m]
            q = e.meanValue
            p = e.priorP
            
            #if self.parent == None: # and explore: #hmmm
            p = 0.8 * p + 0.2 * dirNoise[idx]
            
            u = cpuct * p * (allVisitsSq / (1 + e.visitCount))
            value = q + u
#             foo.append((m, value))
            if (moveName == None or value > moveValue):
                moveName = m
                moveValue = value
#         print(foo)
        selectedEdge = self.edges[moveName]
        if selectedEdge.childNode == None:
            selectedEdge.childNode = self.executeMove(moveName)
        
        return selectedEdge.childNode
    
    def needsExpand(self):
        return len(self.edges) == 0

    def backup(self, vs):
        if self.parent != None:
            self.parent.visitCount += 1
            self.parent.totalValue += vs[self.parent.parentNode.state.getPlayerOnTurnIndex()]
            self.parent.meanValue = self.parent.totalValue / self.parent.visitCount
            self.parent.parentNode.backup(vs)

    '''
    evaluator is an AbstractLearner
    '''
    def expandEvalBack(self, evaluator):
        if self.state.isTerminal():
            vs = [1.0 / self.state.getPlayerCount()] * self.state.getPlayerCount()
            if self.state.getWinner() != -1:
                vs = [0] * self.state.getPlayerCount()
                vs[self.state.getWinner()] = 1.0
        else:
            prevState = None
            if self.parent != None:
                prevState = self.parent.parentNode.state
            movePMap, vs = evaluator.evaluate(self.state, prevState)
            
            for moveName in dict.keys(movePMap):
                self.edges[moveName] = TreeEdge(movePMap[moveName], self)
            self.moveOptions = sorted(dict.keys(self.edges))
#         print("backup", vs)
        self.backup(vs)

class NeuralMctsTrainer():
    # TODO cpuct is a completely blind guess...
    def __init__(self, stateCreator, evaluator, workingdir, trainingRuns = 3, 
                 movesToCheckMin=800, movesToCheckMax=800,
                 moveToCheckIncreasePerIteration=50, cpuct=1.0):
        assert isinstance(evaluator, AbstractLearner)

        self.trainingRuns = trainingRuns
        self.movesToCheckMin = movesToCheckMin
        self.movesToCheckMax = movesToCheckMax
        self.moveToCheckInc = moveToCheckIncreasePerIteration
        self.setMovesToCheck(0)
        self.cpuct = cpuct
        self.evaluator = evaluator
        self.stateCreator = stateCreator
        self.workingdir = workingdir
        self.frameSets = []
        
        self.champion = None
        self.challenger = None
        
    def setMovesToCheck(self, iteration):
        self.movesToCheck = self.movesToCheckMin
        self.movesToCheck += iteration * self.moveToCheckInc
        self.movesToCheck = min(self.movesToCheck, self.movesToCheckMax)
        print("Moves to check is currently " + str(self.movesToCheck))
        
    def pickMove(self, d, state, explore=False):
        ms = []
        ps = []
        for k in dict.keys(d):
            if state.isMoveLegal(k):
                ms.append(k)
                ps.append(d[k])

        assert len(ms) > 0, "The state should have legal moves or be terminal"
        m = None
        if explore:
            m = np.random.choice(ms, p = ps)
        else:
            m = ms[np.argmax(ps)]
        return m
    
    def _findCompetitiveMove(self, gameState, evaluator):
        tree = TreeNode(gameState)
        moveProbs = self.mcts(tree, evaluator=evaluator)
        return self.pickMove(moveProbs, gameState, explore=False)
    
    def findCompetitiveMove(self, gameState):
        tree = TreeNode(gameState)
#         print(self.evaluator.evaluate(gameState, None))
        moveProbs = self.mcts(tree)
        return self.pickMove(moveProbs, gameState, explore=False)
    
    def countNumberOfUniqueStates(self, frames):
        groups = []
        
        for frame in frames:
            state = frame[0]
            known = False
            for g in groups:
                if state.isEqual(g[0]):
                    known = True
                    g.append(state)
            if not known:
                groups.append([state])
        
        return len(groups)
    
    def collectGameFrames(self, explore=True):
        tree = TreeNode(self.stateCreator())
        
        frames = []
        
        while not tree.state.isTerminal():
            moveProbs = self.mcts(tree)
            if tree.state.getTurn() > 0:
                frames.append((tree.state.clone(), moveProbs, [0] * tree.state.getPlayerCount()))
            mv = self.pickMove(moveProbs, tree.state, explore and tree.state.isEarlyGame())
            tree = tree.getChildForMove(mv)
        
        winner = tree.state.getWinner()
        if winner != -1:
            for frame in frames:
                frame[2][winner] = 1
        else:
            for frame in frames:
                for idx in range(len(frame[2])):
                    frame[2][idx] = 1.0 / tree.state.getPlayerCount()
        
        return frames

    def collectNGameFrames(self, n, pid):
        print("Process %i| Collecting frames from %i games..." % (pid, n))
        allframes = []
        for i in range(n):
            t = time.time()
            for f in self.collectGameFrames():
                allframes.append(f)
            if i % 5 == 0:
                print("Process %i| Finished game %i in %f sec" % (pid, i, time.time() - t))
        random.shuffle(allframes)
        return allframes

    '''
    given a starting state tree node return a distribution of probabilities over the moves to take
    in the starting state for approx. optimal play. The caller has to decide if it should exploit (i.e.
    use the highest scoring move), or explore (i.e. try other things)
    '''
    def mcts(self, s0, evaluator=None):
        current = s0
        checkedMoves = 0
        
        eva = evaluator
        if eva == None:
            eva = self.evaluator
        
        stop = False
        
        assert not s0.state.isTerminal()
        
#         print(s0.state.mnk)
        
        while not stop:
            while not current.needsExpand() and not current.state.isTerminal():
                checkedMoves += 1
                current = current.selectMove(self.cpuct)
                
#                 print(current.state.mnk)
                
                if checkedMoves >= self.movesToCheck:
                    stop = True
                    break
                
            if not stop:
                current.expandEvalBack(eva)
#             print("===")
            current = s0
        
        sumv = 0
        for m in dict.keys(s0.edges):
            e = s0.edges[m]
            sumv += e.visitCount
        
        r = dict()
        for m in dict.keys(s0.edges):
            e = s0.edges[m]
            r[m] = e.visitCount / float(sumv)
            
#         print(r)
            
        return r
    
    def loadIteration(self, i):
        f = None
        if i != None:
            f = os.path.join(self.workingdir, "net.iter" + str(i))
        self.evaluator.initState(f)
    
    def demoPlayGames(self, showLearning=False):
        print("==========")
        frames = self.collectGameFrames(explore=False)
        for f in frames:
            print(f[0].mnk, f[1], f[2])
        if showLearning:
            self.evaluator.epochs = 1
            self.evaluator.batchSize = len(frames)
            self.evaluator.learnFromFrames(frames, dbg = True)

    def playChampionMatch(self, champion, challenger, flip):
        a = champion
        b = challenger
        if flip:
            c = a
            a = b
            b = c
        
        state = self.stateCreator()
#             print(flip)
        while not state.isTerminal():
#                 print(state.mnk)
            if state.getTurn() % 2 == 0:
                m = self._findCompetitiveMove(state, a)
            else:
                m = self._findCompetitiveMove(state, b)
            state.simulate(m)
#             print(state.mnk)
#             print("====")
            
        winner = state.getWinner()
#             print(winner)
        gresult = [0,0]
        if winner > -1:
            if flip:
                if winner == 0:
                    gresult[1] = 1
                else:
                    gresult[0] = 1
            else:
                if winner == 0:
                    gresult[0] = 1
                else:
                    gresult[1] = 1
        return gresult

    #TODO this assumes a two player game for now...
    #for FFA style more player games would need to use half/half both player types and sum the wins of the types
    # todo why does mp not work here like it does in collecting games? wtf?
    def isChampionDefeated(self, champion, challenger, pool, games=100):
        championWins = 0
        challengerWins = 0
        
#         self.champion.net.cpu()
#         self.champion.net.cpu()
        
#         if pool == None:
        for g in range(games):
            r = self.playChampionMatch(champion, challenger, g < games / 2)
            championWins += r[0]
            challengerWins += r[1]
#         else:

#         asyncs = []
#         for g in range(games):
#             asyncs.append(pool.apply_async(self.playChampionMatch, args=(champion, challenger, g < games / 2,)))
#         for asy in asyncs:
#             for r in asy.get():
#                 championWins += r[0]
#                 challengerWins += r[1]

#         self.champion.net.cuda()
#         self.challenger.net.cuda()

        print("Champion won %i games, challenger won %i games. Challenger win rate is %f" % 
              (championWins, challengerWins, challengerWins / float(challengerWins+championWins)))
        if challengerWins > 0.55 * (championWins+challengerWins):
            print("A new champion was found. Progress was made.")
            return True
        return False
    

    def iterateLearning(self, iterations, gamesPerIter, startingIteration=0, procs=4, dbgUniqueStates=False):
        f = None
        if startingIteration > 0:
            assert self.workingdir != None
            self.loadIteration(startingIteration-1)
            
        pool = mp.Pool(processes=procs)
        
        self.evaluator.initState(f)
        
        self.learner = self.evaluator.clone()
        
        for i in range(startingIteration, iterations):
            self.setMovesToCheck(i)
            t = time.time()
            print("Iteration %i, collecting games" % i)
            
#             gamesPerProc = int(gamesPerIter / procs)
#             missing = gamesPerIter % procs
#             
#             asyncs = []
#             
#             for pid in range(procs):
#                 g = gamesPerProc
#                 if pid == 0:
#                     g += missing
#                 asyncs.append(pool.apply_async(self.collectNGameFrames, args=(g,pid)))
#             
#             frames = []
#             
#             for asy in asyncs:
#                 for f in asy.get():
#                     frames.append(f)
            
            frames = self.collectNGameFrames(gamesPerIter, 0)
            
            print("%f frames per game" % (len(frames) / float(gamesPerIter)))
            
            self.frameSets.append(frames)
            
            
            lastFrameSetUsed = -1
            winSums = [0] * frames[0][0].getPlayerCount()
            frames = []
            for si in reversed(range(len(self.frameSets))):
                fi = 0
                lastFrameSetUsed = si
                fset = self.frameSets[si]
                random.shuffle(fset)
                while len(frames) < self.evaluator.getFramesPerIteration() and fi < len(fset):
                    frames.append(fset[fi])
                    for i in range(len(fset[fi][2])):
                        winSums[i] += fset[fi][2][i]
                    fi+=1
                
                if len(frames) >= self.evaluator.getFramesPerIteration():
                    break
            
            print("Wins situation in played games is: ", winSums)
            
            if lastFrameSetUsed > 0:
                del self.frameSets[0] 

            if dbgUniqueStates:
                # this is quite slow for larger frame numbers and lots of unique states. Too slow in fact
                print("Counting unique states....")
                un = self.countNumberOfUniqueStates(frames)
                print("Among %i frames there are %i unique game states to learn from" % (len(frames), un))
            
            print("Iteration %i, collecting games took %f, training with %i frames..." % (i, time.time() - t, len(frames)))

            improved = False
            for r in range(self.trainingRuns):
                print("Training run %i/%i ..." % (r+1, self.trainingRuns))
                self.learner.learnFromFrames(frames)
                if self.isChampionDefeated(self.evaluator, self.learner, pool):
                    self.evaluator = self.learner.clone()
                    improved = True
                    
            if improved: #hmmmmmmmmmm
                self.learner = self.evaluator.clone()
            
            if self.workingdir != None and improved:
                self.evaluator.saveState(os.path.join(self.workingdir, "net.iter" + str(i)))
