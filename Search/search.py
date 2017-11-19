# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    #Took reference of Algorithm from book : Artificial Intelligence A Modern Approach Figure 3.7
    #Graph Search Problem
    Start_Node = problem.getStartState()  #get the first starting node
    #stack follows LIFO
    Frontier_DFS = util.Stack()         #initialized the frontier/stack using the initial state of problem from util class
    Visited_List_DFS = list()               #initialized the visited nodes list to blank; keeps track of visited nodes

    Frontier_DFS.push((Start_Node,[]))  #push first node in stack

    while not Frontier_DFS.isEmpty():   #if the frontier is empty then return failure
        #choose a leaf node and remove it from the frontier
        Node, Node_Directions = Frontier_DFS.pop()   #Remove the node from the stack = Node; get the directions of the node = Node_Directions

        if(Node in Visited_List_DFS): #Checks whether the popped node has been visited or not; if it is visited move on to next node
            continue

        if problem.isGoalState(Node):  #Checks whether the popped node is the goal or not
            return Node_Directions      #if the node contains a goal state then return the corresponding solution

        Visited_List_DFS.append(Node)       #add the node to the visited list if traversed firt time

        #expand the chosen node, adding the resulting nodes to the frontier
        Successor_Node_DFS = problem.getSuccessors(Node)  #get the child node of the popped node
        for Child_Node, Child_Directions, Path_Cost in Successor_Node_DFS:
            if(Child_Node in Visited_List_DFS):     #Checks whether successor has been visited or not
                continue
            Frontier_DFS.push((Child_Node, Node_Directions+[Child_Directions]))     #only if not in the frontier or explored set

    return [] #returns failure when frontier/stack is empty

def breadthFirstSearch(problem):        #The algorithm works the same as DFS but the data structure used is different
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    Start_Node = problem.getStartState()        #get the first starting node
    #Queue follows FIFO
    Frontier_BFS = util.Queue()     #initialize the queue using the initial state of problem from util class
    Visited_List_BFS = list()   #initialize the visited nodes list to blank; keeps track of visited nodes

    Frontier_BFS.push((Start_Node,[]))  #push first node in queue

    while not Frontier_BFS.isEmpty():       #if the frontier is empty then return failure

        Node, Node_Directions = Frontier_BFS.pop()
        if(Node in Visited_List_BFS):   #Checks whether the popped node has been visited or not; if it is visited move on to next node
            continue

        if problem.isGoalState(Node):   #Checks whether the popped node is the goal node or not
            return Node_Directions      #if the node contains a goal state then return the corresponding solution

        Visited_List_BFS.append(Node)   #add the node to the visited list if traversed firt time

        Successor_Node_BFS = problem.getSuccessors(Node)
        for Child_Node, Child_Directions, Path_Cost in Successor_Node_BFS:
            if(Child_Node in Visited_List_BFS):     #Checks whether successor has been visited or not
                continue
            Frontier_BFS.push((Child_Node, Node_Directions+[Child_Directions]));    #only if not in the frontier or explored set

    return []   #returns failure when frontier/queue is empty

def uniformCostSearch(problem):     #The algorithm works the same as DFS but the data structure used is different
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    Start_Node = problem.getStartState()
    Frontier_UCS = util.PriorityQueue()     #initialize the priorityqueue using the initial state of problem from util class
    Visited_List_UCS = list()       #initialize the visited nodes list to blank; keeps track of visited nodes

    Frontier_UCS.push((Start_Node,[],0),0)

    while not Frontier_UCS.isEmpty():   #if the frontier is empty then return failure

        Node, Node_Directions, Node_Cost  = Frontier_UCS.pop()

        if(Node in Visited_List_UCS):   #Checks whether the popped node has been visited or not; if it is visited move on to next node
            continue

        if problem.isGoalState(Node):   #Checks whether the popped node is the goal or not
            return Node_Directions       #if the node contains a goal state then return the corresponding solution

        Visited_List_UCS.append(Node);  #add the node to the visited list if traversed first time

        Successor_Node_UCS = problem.getSuccessors(Node)
        for Child_Node, Child_Directions, Path_Cost in Successor_Node_UCS:
            if(Child_Node in Visited_List_UCS):     #Checks whether successor has been visited or not
                continue
            Frontier_UCS.push((Child_Node, Node_Directions+[Child_Directions], Node_Cost+Path_Cost), Node_Cost+Path_Cost) #only if not in the frontier or explored set

    return []     #returns failure when frontier/queue is empty

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #heuristics take search states and return numbers that estimate the cost to a nearest goal
    Start_Node = problem.getStartState()
    Frontier_AS = util.PriorityQueue()      #initialize the priority queue using the initial state of problem from util class
    Visited_List_AS = list()

    Frontier_AS.push((Start_Node,[],0),0)

    while not Frontier_AS.isEmpty():        #if the frontier is empty then return failure

        Node, Node_Directions, Node_Cost = Frontier_AS.pop()

        if(Node in Visited_List_AS):    #Checks whether the popped node has been visited or not; if it is visited move on to next node
            continue

        if problem.isGoalState(Node):   #Checks whether the popped node is the goal or not
            return Node_Directions      #if the node contains a goal state then return the corresponding solution

        Visited_List_AS.append(Node)        #add the node to the visited list if traversed firt time

        Successor_Node_AS = problem.getSuccessors(Node)
        for Child_Node, Child_Directions, Path_Cost in Successor_Node_AS:
            if(Child_Node in Visited_List_AS):  #Checks whether successor has been visited or not
                continue
            Heuristic_Value = heuristic(Child_Node, problem)    #heuristic value is calculated from manhattanHeuristic function defined in searchAgent.py
            Frontier_AS.push((Child_Node, Node_Directions+[Child_Directions], Node_Cost+Path_Cost),Node_Cost+Path_Cost+Heuristic_Value) #only if not in the frontier or explored set

    return []   #returns failure when frontier/queue is empty


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
