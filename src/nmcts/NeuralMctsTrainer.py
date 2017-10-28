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

class NeuralMctsTrainer():
    
    def __init__(self, nplayer, epochRuns, workingdir, championGames=500, batchSize=200, threads=5):
        self.bestPlayer = nplayer
        self.learner = nplayer
        self.epochRuns = epochRuns
        self.workingdir = workingdir
        self.pool = mp.Pool(processes=threads)
        self.threads = threads
        self.batchSize = batchSize
        self.frameSets = []
        self.championGames = championGames
    
    # TODO this still assumes two players
    def learnerIsNewChampion(self):
        gamesPerProc = int(self.championGames / self.threads)
        missing = self.championGames % self.threads
        
        assert missing == 0, str(missing) + " != " + 0
        
        bsize = gamesPerProc 
        
        asyncs = []
        for pid in range(self.threads):
            g = gamesPerProc
            if pid == 0:
                g += missing
            asyncs.append(self.pool.apply_async(self.learner.playAgainst, args=(g, bsize, [self.bestPlayer])))
        
        sumResults = [0,0,0]
        
        for asy in asyncs:
            r, _ = asy.get()
            for i in range(len(r)):
                sumResults[i] += r[i]
        
        myWins = sumResults[0]
        otherWins = sum(sumResults[1:-1])
        print("Learner wins %i, best player wins %i, %i draws" % (myWins, otherWins, sumResults[-1]))
        improved = myWins > (myWins + otherWins) * 0.55
        if improved:
            print("Progress was made")
        return improved
    
    def doLearningIteration(self, games):
        t0 = time.time()
        t = time.time()
        
        gamesPerProc = int(games / self.threads)
        missing = games % self.threads
        
        print("Games per process: " + str(gamesPerProc))
        
        asyncs = []
        
        for pid in range(self.threads):
            g = gamesPerProc
            if pid == 0:
                g += missing
            asyncs.append(self.pool.apply_async(self.bestPlayer.selfPlayNGames, args=(g, self.batchSize)))
            
        frames= []
        
        for asy in asyncs:
            for f in asy.get():
                frames.append(f)
                
        
        self.frameSets.append(frames)
        
        lastFrameSetUsed = -1
        frames = []
        for si in reversed(range(len(self.frameSets))):
            fi = 0
            lastFrameSetUsed = si
            fset = self.frameSets[si]
            random.shuffle(fset)
            while len(frames) < self.learner.learner.getFramesPerIteration() and fi < len(fset):
                frames.append(fset[fi])
                fi+=1
            
            if len(frames) >= self.learner.learner.getFramesPerIteration():
                break
        
        if lastFrameSetUsed > 0:
            del self.frameSets[0]
        
        print("Collected %i games with %i frames in %f" % (games, len(self.frameSets[-1]), (time.time() - t)))
        
        t = time.time()
        for _ in range(self.epochRuns):
            self.learner.learner.learnFromFrames(frames)
            if self.learnerIsNewChampion():
                self.bestPlayer = self.learner.clone()

        print("Done learning in %f" % (time.time() - t))

        print("Iteration completed in %f" % (time.time() - t0))

        #TODO should the learner be reset to the bestplayer here? Or keep the not-so-optimal learning progress?

    def loadForIteration(self, iteration):
        self.bestPlayer.learner.initState(os.path.join(self.workingdir, "bestPlayer.iter" + str(iteration)))
        self.learner.learner.initState(os.path.join(self.workingdir, "learner.iter" + str(iteration)))
        with open(os.path.join(self.workingdir, "frameSets" + str(iteration) + ".pickle"), "rb") as f:
            self.frameSets = pickle.load(f)
            print("Loaded %i framesets " % len(self.frameSets))
        
    def saveForIteration(self, iteration):
        self.bestPlayer.learner.saveState(os.path.join(self.workingdir, "bestPlayer.iter" + str(iteration)))
        self.learner.learner.saveState(os.path.join(self.workingdir, "learner.iter" + str(iteration)))
        with open(os.path.join(self.workingdir, "frameSets"+ str(iteration) +".pickle"), "wb") as f:
            pickle.dump(self.frameSets, f)

    def iterateLearning(self, numGames, numIterations, startAtIteration = 0):
        loadIteration = startAtIteration - 1
        if loadIteration > -1:
            self.loadForIteration(loadIteration)
            
        for i in range(startAtIteration, numIterations):
            print("Begin iteration %i" % i)
            self.doLearningIteration(numGames)
            self.saveForIteration(i)