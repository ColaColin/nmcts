'''
Created on Oct 27, 2017

@author: cclausen
'''

import abc



class AbstractState(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def getWinner(self):
        """
        return index of the winning player or -1 if a draw
        """
    
    @abc.abstractmethod
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
    
    @abc.abstractmethod
    def isEarlyGame(self):
        """
        return if moves should be deterministic(False) or probabilistic (True)
        """
    
    @abc.abstractmethod
    def canTeachSomething(self):
        """
        returns True iff the learner may learn something from this state
        """
    
    @abc.abstractmethod
    def getPlayerCount(self):
        """
        returns the number of players who play this game
        """
    
    @abc.abstractmethod
    def getMoveCount(self):
        """
        returns the number of moves a player can make, including currently illegal moves (TODO: why include that?)
        """
    
    @abc.abstractmethod
    def clone(self):
        """
        returns a deep copy of the state
        """
    
    @abc.abstractmethod
    def getFrameClone(self):
        """
        returns a copy that will be used as data for frames. Can probably save some memory. The returned object
        strictly only needs to implement mapPlayerIndexToTurnRel (TODO that should not be necessary...)
        and be useful to the fillNetworkInput implementation
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
