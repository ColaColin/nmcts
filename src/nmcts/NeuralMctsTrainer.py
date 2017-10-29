'''

A class that takes an AbstractLearner and teaches it to play better using
expert iteration. The actual playing is done by NeuralMctsPlayer

Created on Oct 27, 2017

@author: cclausen
'''

import multiprocessing as mp
import time
import random
import os
import pickle

from nmcts.MctsTree import TreeNode

class NeuralMctsTrainer():
    
    def __init__(self, nplayer, epochRuns, workingdir, championGames=500, batchSize=200, threads=5):
        self.bestPlayer = nplayer.clone()
        self.learner = nplayer
        self.epochRuns = epochRuns
        self.workingdir = workingdir
        self.pool = mp.Pool(processes=threads)
        self.threads = threads
        self.batchSize = batchSize
        self.frameHistory = []
        self.championGames = championGames
    
    # TODO this still assumes two players
    def learnerIsNewChampion(self):
        gamesPerProc = int(self.championGames / self.threads)
        missing = self.championGames % self.threads
        
        assert missing == 0, str(missing) + " != " + 0
        
        asyncs = []
        asyncsInverted = []
        for _ in range(self.threads):
            g = int(gamesPerProc / 2)
            asyncs.append(self.pool.apply_async(self.learner.playAgainst, args=(g, g, [self.bestPlayer])))
            asyncsInverted.append(self.pool.apply_async(self.bestPlayer.playAgainst, args=(g, g, [self.learner])))
        
        sumResults = [0,0,0]
        
        for asy in asyncs:
            r, _ = asy.get()
            for i in range(len(r)):
                sumResults[i] += r[i]
        
        for asy in asyncsInverted:
            r, _ = asy.get()
            sumResults[0] += r[1]
            sumResults[1] += r[0]
            sumResults[2] += r[2]
        
        myWins = sumResults[0]
        otherWins = sum(sumResults[1:-1])
        eps = 0.00000001
        print("Learner wins %i, best player wins %i, %i draws: Winrate of %f" % 
              (myWins, otherWins, sumResults[-1], (myWins + eps) / (myWins + otherWins + eps)))
        improved = myWins > (myWins + otherWins) * 0.55
        if improved:
            print("Progress was made")
        return improved
    
    def updateFrameHistory(self, updateTargets):
        if updateTargets:
            bsize = self.batchSize * 4
            batches = int((len(self.frameHistory) / bsize) + 0.5)
            asyncs = []
            for bi in range(batches):
                asyncs.append(self.pool.apply_async(self.bestPlayer.getBatchMctsResults,
                                      args=(self.frameHistory[bi * bsize : (bi+1) * bsize], bi * bsize)))
            for asy in asyncs:
                for f in asy.get():
                    self.frameHistory[f[0]][1] = f[1]
                    self.frameHistory[f[0]][2] = f[2]
                    
        self.frameHistory.sort(key=lambda f: f[2])
        print("Best win chance in the frames is "+str(self.frameHistory[-1][2]))
        hwin = 0
        rmCount = 0
        while len(self.frameHistory) > self.learner.learner.getFramesPerIteration():
            assert hwin <= self.frameHistory[0][2]
            hwin = self.frameHistory[0][2]
            rmCount += 1
            del self.frameHistory[0]
            
        print("Highest win chance removed was %i, removed %i frames" % (hwin, rmCount))
    
    def doLearningIteration(self, games):
        t = time.time()
        t0 = t
        
        gamesPerProc = int(games / self.threads)
        missing = games % self.threads
        assert missing == 0 #just be careful...
        
        print("Games per process: " + str(gamesPerProc))
        
        asyncs = []
        
        for _ in range(self.threads):
            g = gamesPerProc
            asyncs.append(self.pool.apply_async(self.bestPlayer.selfPlayNGames, args=(g, self.batchSize)))
        
        cframes = 0
        learnFrames = []
        for asy in asyncs:
            for f in asy.get():
                cframes += 1
                self.frameHistory.append(f)
                learnFrames.append(f)
        
        print("Collected %i games with %i frames in %f" % (games, cframes, (time.time() - t)))
        
        random.shuffle(self.frameHistory)
        
        for historicFrame in self.frameHistory:
            if len(learnFrames) < self.learner.learner.getFramesPerIteration():
                learnFrames.append(historicFrame)
            else:
                break
        
        t = time.time()
        runs = self.epochRuns
        improved = False
        while runs > 0:
            runs -= 1
            
            random.shuffle(learnFrames)
            
            self.learner.learner.learnFromFrames(learnFrames)
            if self.learnerIsNewChampion():
                self.bestPlayer = self.learner.clone()
                improved = True
                runs += 1

        print("Done learning in %f" % (time.time() - t))

        tu = time.time()

        print("Updating frame history ...")
        self.updateFrameHistory(improved)
        print("Updates done in %f" % (time.time() - tu))

        print("Iteration completed in %f" % (time.time() - t0))
        
        return improved

        #TODO should the learner be reset to the bestplayer here? Or keep the not-so-optimal learning progress?

    def loadForIteration(self, iteration):
        files = os.listdir(self.workingdir)
        foundBest = False
        
        for i in reversed(range(iteration + 1)):
            p = os.path.join(self.workingdir, "bestPlayer.iter" + str(i))
            if p in files:
                foundBest = True
                break
            
        self.learner.learner.initState(os.path.join(self.workingdir, "learner.iter" + str(iteration)))
        
        if foundBest:
            self.bestPlayer.learner.initState(p)
        else:
            self.bestPlayer = self.learner.clone()
        
        with open(os.path.join(self.workingdir, "frameHistory" + str(iteration) + ".pickle"), "rb") as f:
            self.frameHistory = pickle.load(f)
            print("Loaded %i frames " % len(self.frameHistory))
        
    def saveForIteration(self, iteration, improved):
        if improved:
            self.bestPlayer.learner.saveState(os.path.join(self.workingdir, "bestPlayer.iter" + str(iteration)))
        self.learner.learner.saveState(os.path.join(self.workingdir, "learner.iter" + str(iteration)))
        with open(os.path.join(self.workingdir, "frameHistory"+ str(iteration) +".pickle"), "wb") as f:
            pickle.dump(self.frameHistory, f)

    def iterateLearning(self, numGames, numIterations, startAtIteration = 0):
        loadIteration = startAtIteration - 1
        if loadIteration > -1:
            self.loadForIteration(loadIteration)
            
        for i in range(startAtIteration, numIterations):
            print("Begin iteration %i" % i)
            impr = self.doLearningIteration(numGames)
            self.saveForIteration(i, impr)
            