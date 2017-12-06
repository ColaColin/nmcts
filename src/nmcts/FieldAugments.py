'''
Created on Nov 25, 2017

@author: cclausen
'''

import math
import random

def initField(m, n, scalar):
    result = []
    for _ in range(n):
        result.append([scalar] * m)
    return result

def mirrorField(m, n, field):
    for y in range(n):
        line = field[y]
        for x in range(math.floor(m / 2)):
            tmp = line[x]
            line[x] = line[-1-x]
            line[-1-x] = tmp

def rotateFieldLeft(m, n, field):
    tmp = initField(m, n, 0)
    for y in range(n):
        for x in range(m):
            tmp[y][x] = field[y][x]
            
    for y in range(n):
        for x in range(n):
            field[x][y] = tmp[y][-1-x]

def printField(m, n, field):
    s = ""
    for y in range(n):
        for x in range(m):
            s += "{0:.4f}".format(field[y][x]) + " "
        s += "\n"
    print(s)
    
def augmentFieldAndMovesDistribution(m, n, board, moves, moveIdxToPos, movePosToIdx):
    dbg = False
    
    if dbg:
        print("Pre Augment")
        printField(m, n, board)
        print(moves)

    fStateField = board
    fMovesField = initField(m, n, 0)
    for idx, fMove in enumerate(moves):
        x, y = moveIdxToPos(idx)
        fMovesField[y][x] = fMove
        
    if dbg:
        printField(m, n, fMovesField)
    
    if random.random() > 0.5:
        if dbg:
            print("Do mirror")
        mirrorField(m, n, fStateField)
        mirrorField(m, n, fMovesField)
    
    rotations = random.randint(0,3)
    
    if dbg:
        print("Do %i rotations to the left" % rotations)
    
    for _ in range(rotations):
        rotateFieldLeft(m, n, fStateField)
        rotateFieldLeft(m, n, fMovesField)
        
    for y in range(n):
        for x in range(m):
            moves[movePosToIdx(x, y)] = fMovesField[y][x]
    
    if dbg:
        print("Post Augment")
        printField(m,n,fStateField)
        print(moves)
        for idx, fMove in enumerate(moves):
            x, y = moveIdxToPos(idx)
            fMovesField[y][x] = fMove
        printField(m, n, fMovesField)

    
# m = 4
# n = 4
# f = initField(m, n, 1)
# f[1][1] = 8
# f[1][2] = 8
# f[1][3] = 8
# f[2][3] = 8
# 
# printField(m,n, f)
# rotateFieldLeft(m,n, f)
# printField(m,n, f)
# rotateFieldLeft(m,n, f)
# printField(m,n, f)
# rotateFieldLeft(m,n, f)
# printField(m,n, f)
# rotateFieldLeft(m,n, f)
# printField(m,n, f)