# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        #print successorGameState
        newPos = successorGameState.getPacmanPosition()
        #print newPos
        newFood = successorGameState.getFood()
        #print newFood
        newGhostStates = successorGameState.getGhostStates()
        #print newGhostStates
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print newScaredTimes


        "*** YOUR CODE HERE ***"

        #Find the current food list
        Current_Food = currentGameState.getFood()
        #Making it into a list
        Current_Food_List = Current_Food.asList()

        #currentFood = successorGameState.getFood()
        #print currentFood

        #Find the current State of Ghost from gameState Class
        Current_Ghost_State = currentGameState.getGhostStates()


        Distance_Agent_toFood = 50000000   #Initializing distance to some large finite value to indicate that negative of this value is a bad next state for the agent
        #if currentFood[x][y] == True:

        for X_Ghost_State in newGhostStates:     #For each ghost state in a successor state of ghost finding the position

          Position_of_Ghost = X_Ghost_State.getPosition()

          if Position_of_Ghost == newPos:    #To check if pacman and ghost are on same coordinates it is a bad move and hence just return a very low value
            return -50000000

        #print action
        for Current_Food in Current_Food_List:
            #Finding the minimum distance between the dot/food and pacman's next position
          Distance_Agent_toFood = min(Distance_Agent_toFood,manhattanDistance(Current_Food,newPos))
          #print Distance_Agent_toFood
          if Directions.STOP in action:   #If action has stop then also it is a bad move
            return -50000000

        return 1.0/(1.0 + Distance_Agent_toFood)  #Returns the value of the distance between agent and food as score
        #return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        Dir_Pacman =  self.Find_MiniMaxVal(gameState,0,0)  #Returns the direction of pacman by using MinMax Value
        #Separating out max and min functions
        return Dir_Pacman
        util.raiseNotDefined()

    def Find_MiniMaxVal(self, gameState, Index_of_Agent, Depth_Node):

        #print "Agent Index"

        #print agentIndex

        Tot_Number_Agents = gameState.getNumAgents()   #Find the total number of agents in a gameState

        if Index_of_Agent >= Tot_Number_Agents:     #agentIndex=0 means Pacman, ghosts are >= 1
                Depth_Node  = Depth_Node + 1
                Index_of_Agent = 0      # Pacman is agent 0 and agents move in order of increasing agent index


        #print "Node Depth"
        #print nodeDepth

        if Depth_Node == self.depth:

            return self.evaluationFunction(gameState)       #returns utility for Terminal/ leaf state

        x = self.index

        if Index_of_Agent <> self.index:    ##Checks if the next agent is Min
                return self.Find_MinVal_MM(gameState, Index_of_Agent, Depth_Node)

        else:
                return self.Find_MaxVal_MM(gameState, Index_of_Agent, Depth_Node)

    #Took reference of algorithm from A Modern Approach (3rd Edition) page Figure 5.3 page 166

    def Find_MaxVal_MM(self, gameState, Index_of_Agent, Depth_Node):    #determine backed-up value of a state
     #To initialize max_val to negative infinity
      Maximum_Value = float("-inf")  #v <- -infinity  = Worst case the maximizer

      Actions_Legal = gameState.getLegalActions(Index_of_Agent)
      Value_of_Action = "None"   #initializing actionValue

      if gameState.isWin() or gameState.isLose():       #Checks to see whether it is time to end the game
          #to check if TERMINAL-TEST(state) then return UTILITY(state)

          return self.evaluationFunction(gameState)   #self.evaluationFunction - Finds the utility of game state

      for x in Actions_Legal:   #For each legal action in list of legal acion

        #Find the successor game state after an agent takes an action
          State_Successor = gameState.generateSuccessor(Index_of_Agent,x)    #TO generate next agent

          Node_Val = self.Find_MiniMaxVal(State_Successor,Index_of_Agent+1,Depth_Node)   #Get the MinMax value from function defined above of the next agent

          if Node_Val > Maximum_Value:
              Maximum_Value = max(Node_Val,Maximum_Value)   #Finding the maximum between different nodes in a ply

              Value_of_Action = x

      if Depth_Node == 0:    #Check for leaf node
        return Value_of_Action
      else:
        return Maximum_Value   #return value

    def Find_MinVal_MM(self, gameState, Index_of_Agent, Depth_Node):   #determine backed-up value of a state
    #To initialize max_val to negative infinity
      Minimum_Value = float("inf")   #v <- infinity #v  = Worst case the minimizer
      Actions_Legal = gameState.getLegalActions(Index_of_Agent)
      Value_of_Action = "None"
      Tot_Number_Agents = gameState.getNumAgents()   #Find the total number of agents in a gameState

      if gameState.isWin() or gameState.isLose():       #Checks to see whether it is time to end the game
          #if TERMINAL-TEST(state) then return UTILITY(state)

        return self.evaluationFunction(gameState)

      for x in Actions_Legal:

        State_Successor = gameState.generateSuccessor(Index_of_Agent,x)    #TO generate next agent

        Node_Val = self.Find_MiniMaxVal(State_Successor,Index_of_Agent+1, Depth_Node) #Recursive function #Get the MiniMax value from function defined above of next agent

        if Node_Val < Minimum_Value:
          Minimum_Value = min(Node_Val,Minimum_Value)    #Finding the minimum between different nodes in a ply
          Value_of_Action = x

      return Minimum_Value     #return value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #Alpha is best already explored option along the path to the root for maximizer
        #Beta is the best already explored option along the path to the root for minimizer
        Dir_Pacman = self.Find_AlphaBetaVal(gameState, 0, 0, float("-inf"),float("inf"))    #Returns the direction of pacman by using AplhaBeta function

        return Dir_Pacman
        util.raiseNotDefined()


    def Find_AlphaBetaVal(self, gameState, Index_of_Agent, Depth_Node, Alpha_Val, Beta_Val):

        Tot_Number_Agents = gameState.getNumAgents()   #Find the total number of agents in a gameState
        if Index_of_Agent >= Tot_Number_Agents:

            Depth_Node = Depth_Node + 1
            Index_of_Agent = 0       # Pacman is agent 0 and agents move in order of increasing agent index

        if Depth_Node == self.depth:

                return self.evaluationFunction(gameState)

        if Index_of_Agent <> self.index:        #Checks if the next agent is Min

                return self.Find_MinVal_AB(gameState, Index_of_Agent, Depth_Node, Alpha_Val, Beta_Val)  #Minimizer
        else:

                return self.Find_MaxVal_AB(gameState, Index_of_Agent, Depth_Node, Alpha_Val, Beta_Val)  #Maximizer

    def Find_MaxVal_AB(self, gameState, Index_of_Agent, Depth_Node, Alpha_Val, Beta_Val):   #Reference taken from diagram given in question 3 of berkley AI


      Maximum_Value = float("-inf")  #Initializing worst case value for Maximum_Value/root node also
      Actions_Legal = gameState.getLegalActions(Index_of_Agent)  #Gives all legal actions of agent depending on index of agent
      Value_of_Action = "None"

      if gameState.isWin() or gameState.isLose(): #Checks to see whether it is time to end the game
      #if TERMINAL-TEST(state) then return UTILITY(state)
        return self.evaluationFunction(gameState)

      for x in Actions_Legal:  #For each action in list of legal actions

        State_Successor = gameState.generateSuccessor(Index_of_Agent, x)    #TO generate next agent
        # v = max (v, value(successor, alpha, beta))

        Node_Val = self.Find_AlphaBetaVal(State_Successor, Index_of_Agent+1, Depth_Node, Alpha_Val, Beta_Val) #Finds alphabeta_value value of next agent

        if Node_Val > Maximum_Value:
                Maximum_Value = Node_Val
                Value_of_Action = x

        if Maximum_Value > Beta_Val:     #to check for pruning  #if v > Beta
          return Maximum_Value                                  #return v

        Alpha_Val = max(Alpha_Val, Maximum_Value)    #alpha = max (alpha, v)

      if Depth_Node == 0:   #Check for leaf node

        return Value_of_Action
      else:

        return Maximum_Value   #return v

    def Find_MinVal_AB(self, gameState, Index_of_Agent, Depth_Node, Alpha_Val, Beta_Val): #Reference taken from diagram given in question 3 of berkley AI

      Minimum_Value = float("inf")   #Initializing worst case value for Minimum_Value
      Actions_Legal = gameState.getLegalActions(Index_of_Agent) #Gives all legal actions of agent depending on index of agent
      Value_of_Action = "None"  #initializing actionValue

      if gameState.isWin() or gameState.isLose():      #If the state is a terminal state
          #Checks to see whether it is time to end the game
          #if TERMINAL-TEST(state) then return UTILITY(state)
          return self.evaluationFunction(gameState)

      for x in Actions_Legal:       #For each action in list of legal actions

        State_Successor = gameState.generateSuccessor(Index_of_Agent, x)   #TO generate next agent state
        Node_Val = self.Find_AlphaBetaVal(State_Successor, Index_of_Agent + 1, Depth_Node, Alpha_Val, Beta_Val) #Recursive function #Finds alphabeta_value value of next agent

        if Node_Val < Minimum_Value:

            Minimum_Value = Node_Val
            Value_of_Action = x

        if Minimum_Value < Alpha_Val:     #to check for pruning  #if v < Alpha
          return Minimum_Value                                   #return v

        Beta_Val = min (Beta_Val,Minimum_Value)     #beta = min(alpha, v)

      return Minimum_Value   #return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.Find_ExpectimaxVal(gameState,0,0)
        util.raiseNotDefined()

    def Find_ExpectimaxVal(self, gameState, Index_of_Agent, Depth_Node):

        Tot_Number_Agents = gameState.getNumAgents()   #Find the total number of agents in a gameState
        if Index_of_Agent >= Tot_Number_Agents:

            Depth_Node = Depth_Node + 1
            Index_of_Agent = 0       # Pacman is agent 0 and agents move in order of increasing agent index

        if Depth_Node == self.depth:
            return self.evaluationFunction(gameState)

        if Index_of_Agent <> self.index:    ##Checks if the next agent is Chance Node
                return self.Find_Exp_Val(gameState, Index_of_Agent, Depth_Node)
        else:

                return self.Find_MaxVal_EA(gameState, Index_of_Agent, Depth_Node)


    def Find_MaxVal_EA(self, gameState, Index_of_Agent, Depth_Node):

      Maximum_Value = float("-inf")   #Initializing value for Maximum_Value
      Actions_Legal = gameState.getLegalActions(Index_of_Agent) #Gives all legal actions of agent depending on index of agent
      Value_of_Action = "None"  #initializing actionValue

      if gameState.isWin() or gameState.isLose():  #If the state is a terminal state
          #Checks to see whether it is time to end the game
          #if TERMINAL-TEST(state) then return UTILITY(state)
            return self.evaluationFunction(gameState)

      for x in Actions_Legal:   #For each action in all legal actions

        State_Successor = gameState.generateSuccessor(Index_of_Agent, x)    #TO generate next agent state

        Node_Val = self.Find_ExpectimaxVal(State_Successor, Index_of_Agent+1, Depth_Node)     #Recursive Call   #Finds expectimax_value value of next agent

        if Node_Val > Maximum_Value:
                Maximum_Value = Node_Val
                Value_of_Action = x

      if Depth_Node == 0:   #Check for leaf node

        return Value_of_Action
      else:

        return Maximum_Value   #return v

    def Find_Exp_Val(self, gameState, Index_of_Agent, Depth_Node):

      Expectimax_Value = 0      #initialize expectimax value to 0
      Actions_Legal = gameState.getLegalActions(Index_of_Agent) #Gives all legal actions of agent depending on index of agent


      if gameState.isWin() or gameState.isLose():  #If the state is a terminal state
          #Checks to see whether it is time to end the game
          #if TERMINAL-TEST(state) then return UTILITY(state)
            return self.evaluationFunction(gameState)

      Average_Value = 1.0/len(Actions_Legal)   #Find average at chance node in float as asked in question

      for x in Actions_Legal:    #For each action in leagal actions

        State_Successor = gameState.generateSuccessor(Index_of_Agent, x)    #TO generate next agent state

        Node_Val = self.Find_ExpectimaxVal(State_Successor, Index_of_Agent+1, Depth_Node)     #Recursive Call   #Finds expectimax_value value of next agent

        Expectimax_Value = Expectimax_Value + (Node_Val * Average_Value)   #v=v+p * value(successsor)

      return Expectimax_Value

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      Find the distance of food dot from pacman to decide where dot is closer and then return that value so that
      my score is the best

    """
    "*** YOUR CODE HERE ***"
    Game_Score = currentGameState.getScore()  # Return score of game sstate
    Position_FoodDot = currentGameState.getFood()  #Find position of food
    Position_FoodDot_List = Position_FoodDot.asList()    #Putting position of food in a list for traverse
    Distance_FoodDot = []  #Initializing empty distance of food dot from pacman
    Position_Pacman = currentGameState.getPacmanPosition()   #Find Pacman Position
    Position_Pacman_List = list(Position_Pacman)

    for x in Position_FoodDot_List:  #For each food dot
        #Find manhattanDistance between food dot and pacman
        Distance_Pacman = manhattanDistance(x, Position_Pacman_List)  #calculated manhattanDistance between each food dot and pacman current position
        Distance_FoodDot.append(-1 * Distance_Pacman)   #Used for calculating score

    if not Distance_FoodDot:
        Distance_FoodDot.append(0)

    Max_Distance_FoodDot = max(Distance_FoodDot)   #Find the closest food dot
    betterEvaluationValue = Max_Distance_FoodDot +  Game_Score   #Calculates best value to return
    return  betterEvaluationValue

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
