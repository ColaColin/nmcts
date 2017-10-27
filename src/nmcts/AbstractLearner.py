'''
Created on Oct 27, 2017

@author: cclausen
'''

import abc

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
    def getBatchSize(self):
        """
        """
    
    @abc.abstractmethod
    def evaluate(self, batch):
        """
        gets a list of states to be evaluated
        returns a list of pairs (moveProbability, winChance). The index of the pairs is their moveKey
        this has to be able to deal with None values in the batch! 
        """
        
    @abc.abstractmethod
    def clone(self):
        """
        returns a copy of the evaluator, to be used as a checkpoint or comparison
        """
        
    @abc.abstractmethod
    def learnFromFrames(self, frames, dbg=False):
        """
        learn from the given frames. Each frame is a triple of
        state, moveP, winP
        where moveP and winP are the learning targets for state
        """
        
    def getFramesPerIteration(self):
        """
        returns the maximum number of frames learned from per iteration
        """