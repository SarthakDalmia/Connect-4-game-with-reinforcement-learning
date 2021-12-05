

import math
import random
import enum
import numpy as np
import gzip
import matplotlib.pyplot as plt
import pickle
from IPython.display import clear_output
import ast

def PrintGrid(positions):
    print('\n'.join(' '.join(str(x) for x in row) for row in positions))
    print()

class Result(enum.Enum):
    NonLeaf = 2
    Win =1
    Draw =0
    Loss =-1
        
def checkVertical(grid, startCol):
    col = startCol
    row = 0
    firstPlayer = 0
    secondPlayer = 0
    while  row >= 0 and row < grid.shape[0]:
        if grid[row][col]==1:
            firstPlayer+=1
            secondPlayer=0
        
        elif grid[row][col]==-1:
            firstPlayer=0
            secondPlayer+=1
        else:
            firstPlayer=0
            secondPlayer=0
        if secondPlayer >= 4:
            return Result.Loss
        if firstPlayer >= 4:
            return Result.Win

        row+=1
    
    return Result.Draw

def checkHorizontal(grid, startRow):
    col = 0    
    row = startRow

    firstPlayer = 0
    secondPlayer = 0

    while  col >=0 and col < grid.shape[1]:
        if grid[row][col]==1:
            firstPlayer+=1
            secondPlayer=0
        
        elif grid[row][col]==-1:
            firstPlayer=0
            secondPlayer+=1
        else:
            firstPlayer=0
            secondPlayer=0
        if secondPlayer >= 4:
            return Result.Loss
        if firstPlayer >= 4:
            return Result.Win

        col+=1
    
    return Result.Draw

def checkDiagonal(grid, rowUpdate, colUpdate, startRow, startCol):
    row = startRow
    col = startCol
    firstPlayer = 0
    secondPlayer = 0
    while  col >=0 and row >= 0 and col < grid.shape[1] and row < grid.shape[0]:
        if grid[row][col]==1:
            firstPlayer+=1
            secondPlayer=0
            
        
        elif grid[row][col]==-1:
            firstPlayer=0
            secondPlayer+=1
            
        else:
            firstPlayer=0
            secondPlayer=0
            
        if secondPlayer >= 4:
            return Result.Loss
        if firstPlayer >= 4:
            return Result.Win

        col+=colUpdate
        row+=rowUpdate

    return Result.Draw

def childCreator(grid, move):
    last_index = None
    for row in range(grid.shape[0]):
        if grid[row][move] != 0:
            break
        last_index = row
    
    if last_index != None:
        child_grid = grid.copy()
        child_grid[last_index][move] = 1
        child_grid = -1*child_grid
        return child_grid

def winOrLoss(grid):
    # Calculate state if not win or loss
    result = Result.Draw
    
    for row in range(grid.shape[0]):
        tempResult = checkHorizontal(grid,row)
        if tempResult != Result.Draw:
            return tempResult
            
    for diags in range(10):
        col = max(0,diags-5)
        row = min(5, diags)
        tempResult = checkDiagonal(grid, -1, 1, row, col)
        if tempResult != Result.Draw:
            return tempResult

    for col in range(grid.shape[1]):
        tempResult = checkVertical(grid, col)
        if tempResult != Result.Draw:
            return tempResult
    
    for diags in range(10):
        col = max(0,diags-5)
        row = max(0, 5-diags)
        tempResult = checkDiagonal(grid, 1, 1, row, col)
        if tempResult != Result.Draw:
            return tempResult


    for col in range(grid.shape[1]):
        if grid[0][col] == 0:
            return Result.NonLeaf

    return result

def getPossibleMoves(grid):
    possibleMoves = []
    for col in range(grid.shape[1]):
        if grid[0,col] == 0:
            possibleMoves.append(col)
    return np.array(possibleMoves).ravel()

def to_immutable(grid):
    return tuple(map(tuple, grid))

class Node:
    
    def __init__(self, board, parent, uctConstant):
        self.board = board
        self.parent = parent
        self.uctConstant = uctConstant
        self.children = {}
        self.rewards = 0
        self.plays = 0
    def childSelector(self):
        bestUctValue = -math.inf
        for child in self.children.values():
            if bestUctValue < child.uctValue():
                bestUctValue = child.uctValue()
        childList = []
        for child in self.children.values():
            if(child.uctValue()==bestUctValue):
                childList.append(child)
        return random.choice(childList)

    def registerResult(self, result):
        self.plays += 1
        if result == Result.Loss:
            self.rewards += -1
        elif result == Result.Win:
            self.rewards += 1
    
    def hasChild(self):
        return bool(self.children)

    def findChilds(self):
        self.children = {}
        for move in range(self.board.shape[1]):
            child = childCreator(self.board, move)
            if(child is not None):
                self.children[move] = Node(child, self, self.uctConstant)

    def getResult(self):
        return winOrLoss(self.board)
    
    def uctValue(self):
        if  self.plays != 0 and self.parent != None:
            temp = self.rewards/self.plays
            temp += (self.uctConstant * math.sqrt(math.log(self.parent.plays)/self.plays))
            return temp
        elif  self.plays == 0 and self.parent != None:
            return math.inf

class MCAgent:
    def __init__(self, numSims, board = np.zeros((6,5)).astype(int), rewardMap = { Result.Win:1, Result.Draw:-1, Result.Loss:-1, Result.NonLeaf:0}, uctConst = 1.4):
        self.numSims = numSims
        self.root = Node(board, None, uctConst)
        self.uctConst = uctConst
        self.firstMove = True

    def reset(self, board = np.zeros((6,5)).astype(int)):
        self.root = Node(board, None, self.uctConst)
        self.firstMove = True

    def mover(self):
        
        for sim in range(self.numSims):
            currNode, depth = self.selector()
            
            if not self.firstMove or depth<4:
              currNode = self.expand(currNode=currNode)
            
            result = self.simulate(currBoard=currNode.board)
            self.backPropogation(currNode=currNode, result=result)

        maxPlays = -1
        bestMove = None
        for move in self.root.children:
            if self.root.children[move].plays > maxPlays:
                bestMove = move
                maxPlays = self.root.children[move].plays
                
        bestValue = self.root.children[bestMove].uctValue()
        self.firstMove = False
        self.moveReg(bestMove)
        return bestMove, bestValue

    def selector(self):
        currNode = self.root
        depth = 0
        
        while currNode.getResult()==Result.NonLeaf and currNode.hasChild():
            currNode = currNode.childSelector()
            depth+=1
        return currNode, depth

    def simulate(self, currBoard):
        if winOrLoss(currBoard) == Result.NonLeaf:
            actions = getPossibleMoves(currBoard)
            nextBoard = childCreator(currBoard, random.choice(actions))

            result = self.simulate(nextBoard)
            if result == Result.Loss:
                return Result.Win
            elif result == Result.Win:
                return Result.Loss
            else:
                return result
            
        else:
            result = winOrLoss(currBoard)
            if result == Result.Loss:
                return Result.Win
            elif result == Result.Win:
                return Result.Loss
            else:
                return result


    def moveReg(self, move):
        if not self.root.hasChild():
            self.root.findChilds()
        self.root = self.root.children[move]
        self.root.parent = None

    def backPropogation(self, currNode, result):
        while currNode!=None:
            currNode.registerResult(result)
            if result == Result.Loss:
                result= Result.Win
            elif result == Result.Win:
                result= Result.Loss
            else:
                result= result
            currNode = currNode.parent
    
    def expand(self, currNode):
        if currNode.getResult() == Result.NonLeaf:
            currNode.findChilds()
            return currNode.childSelector()
        else:
            return currNode
    

class AfterstatesAgent:
    
    def __init__(self, board = np.zeros((4,5)).astype(int), rewardMap={Result.Draw:-1, Result.Win:1, Result.Loss:-1 ,Result.NonLeaf:0}, epsilon=0.1, alpha=1, gamma=0.5 ):
        self.currBoard = board
        self.oldBoard = None
        self.valueMap = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma


    def mover(self):

        nextBoard, move, maxValuePossible = self.selectBoard()

        if not self.oldBoard is None:
            result = winOrLoss(nextBoard)
            if result == Result.Win:
                result= Result.Loss
            elif result == Result.Loss:
                result= Result.Win
            else:
                result= result
            if result == Result.Loss:
                reward = -1
            elif result == Result.Win:
                reward = 1
            elif result == Result.Draw:
                reward = -1
            else:
                reward = 0
            self.valueMap[to_immutable(self.oldBoard)] = (1-self.alpha)*self.valueMap.get(to_immutable(self.oldBoard),0) + self.alpha * (reward + self.gamma*maxValuePossible)

        self.currBoard = nextBoard

        return move, self.valueMap.get(to_immutable(nextBoard),0)
    
    def selectBoard(self):
        possibleMoves = getPossibleMoves(self.currBoard)
        possibleBoards = {move:childCreator(self.currBoard, move) for move in possibleMoves}

        maxQ = -math.inf
        for aBoard in possibleBoards.values():
            if maxQ < self.valueMap.get(to_immutable(aBoard),0):
                maxQ = self.valueMap.get(to_immutable(aBoard),0)
        
        greedyMove = -1
        for aMove,aBoard in possibleBoards.items():
            if self.valueMap.get(to_immutable(aBoard),0)==maxQ:
                greedyMove = aMove
                break
        
        if random.random() >= 1-self.epsilon:
            randomMove = possibleMoves[random.randrange(len(possibleMoves))]
            return possibleBoards[randomMove], randomMove, maxQ
        else:
            return possibleBoards[greedyMove], greedyMove, maxQ

    def moveReg(self, move):
        self.oldBoard = self.currBoard
        self.currBoard = childCreator(self.currBoard, move)
        if winOrLoss(self.currBoard)!=Result.NonLeaf:
            if not self.oldBoard is None:
                if winOrLoss(self.currBoard) == Result.Loss:
                    self.valueMap[to_immutable(self.oldBoard)] = -1
                elif winOrLoss(self.currBoard) == Result.Win:
                    self.valueMap[to_immutable(self.oldBoard)] = 1
                elif winOrLoss(self.currBoard) == Result.Draw:
                    self.valueMap[to_immutable(self.oldBoard)] = -1

    def saveVMap(self, filename):
        sfile = gzip.GzipFile(filename, 'w')
        pickle.dump(self.valueMap, sfile)

    def loadVMap(self, filename):
        sfile = gzip.GzipFile(filename, 'r')
        self.valueMap = pickle.load(sfile)

    def reset(self, board = np.zeros((4,5)).astype(int)):
        self.currBoard = board
        self.oldBoard = None


def getAgentName(agent):
    if isinstance(agent, AfterstatesAgent):
        return "Q-Learning"
    else:
        return "MCTS"

def play(board, firstAgent, secondAgent):

    print("**** New Game *****")

    presentAIndex = 0
    numMoves = 0
    secondAgent.reset(board)
    firstAgent.reset(board)
    
    
    agents = [firstAgent, secondAgent]

    while winOrLoss(board) == Result.NonLeaf:
        numMoves+=1

        move, value = agents[presentAIndex].mover()
        agents[1-presentAIndex].moveReg(move)

        print("Player", presentAIndex+1, "("+getAgentName(agents[presentAIndex])+")")
        print("Action Selected :", move)
        print("Value of next state according to", getAgentName(agents[presentAIndex]), ":", "{:.4f}".format(round(value,ndigits=4)))
        
        board = childCreator(board, move)
        
        if presentAIndex != 0:
            printArray = board
        else:
            printArray = -1 * board

        PrintGrid(np.where(printArray == -1, 2, printArray))
        
        presentAIndex = 1-presentAIndex

    print("Player", str(presentAIndex+1), "has", str(winOrLoss(board))+".", "Total moves =", str(numMoves)+".")

def train(MCAgent, gamma, rewardMap, path, trainIters=10000, valIters=200):
    afterAgent = AfterstatesAgent(epsilon=1, rewardMap=rewardMap, gamma=gamma)
    # afterAgent.loadVMap("2018B3A70290G_SARTHAK.dat.gz")
    for epsilon in np.linspace(1, 0.1, 19):
        afterAgent.epsilon = epsilon
        for i in range(trainIters):
            play(np.zeros((4,5)).astype(int),MCAgent, afterAgent)
        afterAgent.epsilon = 0
        for i in range(valIters):
            play(np.zeros((4,5)).astype(int),MCAgent, afterAgent)

    for epsilon in [0.1, 0.075 , 0.05, 0.025, 0.01, 0.005]:
        afterAgent.epsilon = epsilon
        for i in range(trainIters):
            play(np.zeros((4,5)).astype(int),MCAgent, afterAgent)

        afterAgent.epsilon = 0
        for i in range(valIters):
            play(np.zeros((4,5)).astype(int),MCAgent, afterAgent)
    
    afterAgent.saveVMap("2018B3A70290G_SARTHAK.dat.gz")

gamma = 1
rewardMap = {Result.Draw:-0.5, Result.Loss:-1, Result.Win:1 ,Result.NonLeaf:-0.1}
# train(MCAgent(20, uctConst = 1.4, rewardMap={Result.Draw:-1, Result.Loss:-1, Result.Win:1 ,Result.NonLeaf:0}), gamma, rewardMap, "Final/", )

print("1 for the MCTS ouptput (part (a)) and 2 for the qlearning output (part (b))")
x=int(input("Enter the required output: "))
if(x==1):
    play(np.zeros((6,5)).astype(int), MCAgent(200), MCAgent(40))
elif(x==2):
    afterAgent = AfterstatesAgent(np.zeros((4,5)).astype(int), epsilon=0, gamma=1);
    afterAgent.loadVMap("2018B3A70290G_SARTHAK.dat.gz")
    play(np.zeros((4,5)).astype(int), MCAgent(25, board = np.zeros((4,5)).astype(int)), afterAgent)
else:
    print("Invalid Input")