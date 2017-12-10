'''

A class that takes an AbstractLearner and teaches it to play better using
expert iteration. The actual playing is done by NeuralMctsPlayer

Created on Oct 27, 2017

@author: cclausen
'''

import multiprocessing as mp
#import torch.multiprocessing as mp
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
    
    # given an N player game:
    # assert N % 2 == 0 #because I don't want to think about the more general case...
    # setup N/2 players playing as the learner, N/2 as bestPlayer
    # sum up all wins of the learner instances and all wins of the bestPlayer instances
    def learnerIsNewChampion(self):
        gamesPerProc = int(self.championGames / self.threads)
        
        assert gamesPerProc % 2 == 0, "championgames / threads / 2 needs to be even!"
        
        missing = self.championGames % self.threads
        
        assert missing == 0, str(missing) + " != " + 0
        
        playersN = self.learner.stateTemplate.getPlayerCount()
        assert playersN % 2 == 0, "an uneven player count would create more issues and need a bit of code improvement here..."
        
        bestPlayers = int(playersN / 2)
        learners = int(playersN / 2)
        
        dbgFrames = False
        
        asyncs = []
        asyncsInverted = []
        for _ in range(self.threads):
            g = int(gamesPerProc / 2)
            asyncs.append(self.pool.apply_async(self.learner.playAgainst, 
                args=(g, g, [self.learner] * (learners - 1) + [self.bestPlayer] * bestPlayers, dbgFrames)))
            asyncsInverted.append(self.pool.apply_async(self.bestPlayer.playAgainst, 
                args=(g, g, [self.bestPlayer] * (bestPlayers - 1) + [self.learner] * learners, dbgFrames)))
        
        sumResults = [0,0,0]
        
        firstWins = 0
        
        for asy in asyncs:
            r, gframes = asy.get()
            
            if dbgFrames:
                for f in gframes[0]:
                    print(f[0].c6)
                    print(list(reversed(sorted(f[1])))[:5], f[1])
                    print(f[3])
                    print("...")
                
            sumResults[2] += r[-1]
            firstWins += r[0]
            for i in range(len(r)-1):
                if i < learners:
                    sumResults[0] += r[i]
                else:
                    sumResults[1] += r[i]
        
        for asy in asyncsInverted:
            r, gframes = asy.get()
            
            if dbgFrames:
                for f in gframes[0]:
                    print(f[0].c6)
                    print(list(reversed(sorted(f[1])))[:5], f[1])
                    print(f[3])
                    print("...")
            
            sumResults[2] += r[-1]
            firstWins += r[0]
            for i in range(len(r)-1):
                if i < bestPlayers:
                    sumResults[1] += r[i]
                else:
                    sumResults[0] += r[i]
        
        assert sum(sumResults) == self.championGames
        
        myWins = sumResults[0]
        otherWins = sumResults[1]
        eps = 0.00000001
        print("Learner wins %i, best player wins %i, %i draws, %i first move wins: Winrate of %f" % 
              (myWins, otherWins, sumResults[-1], firstWins, (myWins + eps) / (myWins + otherWins + eps)))
        
        improved = myWins > (myWins + otherWins) * 0.55
        if improved:
            print("Progress was made")
            
        return improved
    
    def updateFrameHistory(self, updateTargets):
        doTheFancyStuff = False
        if doTheFancyStuff:
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
            mwc = 0
            for f in self.frameHistory:
                mwc += f[2]
            mwc = float(mwc) / len(self.frameHistory)
            print("Mean win chance in the frames is "+str(mwc))
            hwin = 0
            rmCount = 0
            
            i = 0
            while i < len(self.frameHistory) and len(self.frameHistory) > self.learner.learner.getFramesPerIteration():
                assert hwin <= self.frameHistory[i][2]
                if 0 == self.frameHistory[i][3][self.frameHistory[i][0].getPlayerOnTurnIndex()] and self.frameHistory[i][2] < 0.5:
                    hwin = self.frameHistory[i][2]
                    rmCount += 1
                    del self.frameHistory[i]
                    i -= 1
                    
                i += 1
            
            # TODO why is hwin always zero? Why are there so many frames with a zero win chance?!
            # probably because it was formatted as integer...
            print("Highest win chance removed was %f, removed %i frames by correct low win chance assignment" % (hwin, rmCount))
        # end of fancy stuff that probably is a bad idea...
        
        while len(self.frameHistory) > self.learner.learner.getFramesPerIteration():
            del self.frameHistory[0]
    
    def doLearningIteration(self, games, iteration, keepFramesPerc=1.0):
        t = time.time()
        t0 = t
        
        maxFrames = int(self.learner.learner.getFramesPerIteration() / 3)
        
        framesPerProc = int(maxFrames / self.threads)
        
        print("Frames per process: " + str(framesPerProc))
        
        asyncs = []
        
        for _ in range(self.threads):
#             g = gamesPerProc
#             asyncs.append(self.pool.apply_async(self.bestPlayer.selfPlayNGames, args=(g, self.batchSize, keepFramesPerc)))
            asyncs.append(self.pool.apply_async(self.bestPlayer.selfPlayGamesAsTree, args=(framesPerProc, self.batchSize)))
        
        cframes = 0
        ignoreFrames = 0
        learnFrames = []
        newFrames = []
        for asy in asyncs:
            for f in asy.get():
                cframes += 1
                if cframes < maxFrames:
                    learnFrames.append(f)
                
                newFrames.append(f)
        
        print("Collected %i games with %i frames in %f" % (games, cframes, (time.time() - t)))
        
        # TODO make an abstracted method to ignore frames for reasons unknown to this code
        if ignoreFrames > 0:
            print("Ignored %i frames" % ignoreFrames)
        
        random.shuffle(self.frameHistory)
        
        for historicFrame in self.frameHistory:
            if len(learnFrames) < self.learner.learner.getFramesPerIteration():
                learnFrames.append(historicFrame)
            else:
                break
            
        for f in newFrames:
            self.frameHistory.append(f)
        
        while len(self.frameHistory) > self.learner.learner.getFramesPerIteration():
            del self.frameHistory[0]

        self.saveFrames(iteration)
        
        improved = self.learnFrames(learnFrames, iteration)

#         tu = time.time()
#         print("Updating frame history ...")
#         self.updateFrameHistory(improved)
#         print("Updates done in %f" % (time.time() - tu))

        print("Iteration completed in %f" % (time.time() - t0))
        
        return improved

    def learnFrames(self, learnFrames, iteration):
        t = time.time()
        runs = self.epochRuns
        improved = False
        while runs > 0:
            runs -= 1
            
            random.shuffle(learnFrames)
            
            self.learner.learner.learnFromFrames(learnFrames, iteration)
            # TODO this plays a lot of games. Maybe those games could also be used to learn from? At least the moves of the current best player?
            if self.learnerIsNewChampion():
                self.bestPlayer = self.learner.clone()
                improved = True
                runs += 1
        #TODO should the learner be reset to the bestplayer here? Or keep the not-so-optimal learning progress?
        print("Done learning in %f" % (time.time() - t))
        return improved

    def loadForIteration(self, iteration):
        files = os.listdir(self.workingdir)
        foundBest = False
        for i in reversed(range(iteration + 1)):
            bpi = "bestPlayer.iter" + str(i)
            if bpi in files:
                p = os.path.join(self.workingdir, bpi)
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
    
    def saveFrames(self, iteration):
        with open(os.path.join(self.workingdir, "frameHistory"+ str(iteration) +".pickle"), "wb") as f:
            pickle.dump(self.frameHistory, f)
            print("Saved %i frames for iteration %i" % (len(self.frameHistory), iteration))
    
    def saveForIteration(self, iteration, improved):
        if improved:
            self.bestPlayer.learner.saveState(os.path.join(self.workingdir, "bestPlayer.iter" + str(iteration)))
        self.learner.learner.saveState(os.path.join(self.workingdir, "learner.iter" + str(iteration)))

    def handlePreload(self, preloadFrames):
        if preloadFrames != None:
            with open(os.path.join(self.workingdir, preloadFrames + ".pickle"), "rb") as f:
                preframes = pickle.load(f)
                print("Preloaded %i frames" % len(preframes))
                
#                 c = 0
#                 for f in preframes:
#                     if f[3][0] > 0.00001 and f[3][0] < 0.999999:
#                         c += 1
#                 print(c)
#                 for f in preframes[8090:8100]:
#                     print(f[0].c6)
#                     print(f[1])
#                     print(f[3])
#                     print(",,,")
#                     
#                 exit(0)
                
                for f in preframes:
                    self.frameHistory.append(f)
                self.learnFrames(self.frameHistory, 0)
                
                
    # TODO preload does not store a bestPlayer. Which is not optimal at all...
    def iterateLearning(self, numIterations, numGames, preloadFrames=None, startAtIteration = 0,keepFramesPerc=1.0):
        loadIteration = startAtIteration - 1
        if loadIteration > -1:
            self.loadForIteration(loadIteration)
            
        
        self.handlePreload(preloadFrames)
        
        for i in range(startAtIteration, numIterations):
            print("Begin iteration %i" % i)
            impr = self.doLearningIteration(numGames, i, keepFramesPerc=keepFramesPerc)
            self.saveForIteration(i, impr)
            