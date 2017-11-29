# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        no_of_iterations = self.iterations  #to loop till number of iterations

        for i in range(0,no_of_iterations):

          counter_Dictionary = util.Counter()     #initialize dictionary counter which at the end gives us the final best value iterations
          get_States = mdp.getStates()      #to get all the nodes/coordinates of gridworld

          for action_state in get_States:   #iterate each node of gridworld

                if mdp.isTerminal(action_state):        #Checks if node is terminal state or note

                      counter_Dictionary[action_state] = 0      #assign 0 as value iteration to terminal state
                      continue;                                 #next

                possible_State_Actions = mdp.getPossibleActions(action_state)       #get all the possible action('north','south','east','west') where the agent can go from this state

                valueIteration = 0        #initialize value iteration to 0 for each action
                best_ValueIteration = float('-50000000000')     #initialize to a high value every time to compare between the better Qvalue

                for action in possible_State_Actions:  #looping for each action of state(north, south, east and west)

                  valueIteration = self.getQValue(action_state,action);   #get the Qvalue of the node in gridworld


                  if valueIteration > best_ValueIteration:   #to put in dictionary only the best value
                    best_ValueIteration = valueIteration            #overwriting Qvalue if it is a better Qvalue of previous action
                    counter_Dictionary[action_state] = best_ValueIteration     #for each state storing the best Qvalue

          self.values = counter_Dictionary  #setting dictionary of best value of each node in gridworld


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Q_Value = 0
        gamma = self.discount   #to get the discount factor
        #print "value of state passed as argument is"
        #print state

        #print "value of action passed as argument is"
        #print action

        for next_State, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state,action,next_State)   #to get the reward of state of gridworld
            value = self.getValue(next_State)       #to get the value from the constructor
            Q_Value = Q_Value + probability * ( reward + gamma * value)  #Refer readme for this formula

        return Q_Value      #Returning the QValue

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        policy = None   #initialize best action to take

        valueIteration = 0
        best_ValueIteration = float('-50000000000')

        possible_State_Actions = self.mdp.getPossibleActions(state)     #Get all possible action (exit, north, south, east and west) of state passed as argument

        valueIteration = 0

        for action in possible_State_Actions:

          valueIteration = self.getQValue(state,action);   #Gets the QValue from computeQValueFromValues function for each action {north, south, east and west} of a state


          if valueIteration >= best_ValueIteration:
            policy = action
            best_ValueIteration = valueIteration

        return policy
        
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
