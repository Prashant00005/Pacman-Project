# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.Q = util.Counter()     #initializing Q-Values here

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        Q_Value = self.Q

        if (state,action) not in Q_Value:       #Checks if state is present in dictionary or not, basically if we have ever seen a state or not

              Q_Value[(state,action)]= 0.0  #Put value as zero because we have not seen the state as if now


        return Q_Value[(state,action)]  #Returns Q node value
        #util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        self.Temporary_QValue = util.Counter()  #initializing a temporary QValue counter

        temporary_QValue = self.Temporary_QValue

        maxAction_OverLegalAction = self.getPolicy(state)         #Calls get poilcy which in turn calls the computeActionFromQValues function to get the action we need to take

        if maxAction_OverLegalAction == 0:              #checks if returned state is terminal state
                return 0.0

        temporary_QValue[maxAction_OverLegalAction] = self.getQValue(state,maxAction_OverLegalAction)  #to get the Qvalue of the action returned from computeActionFromQValues function


        return temporary_QValue[maxAction_OverLegalAction]  #Returns the max_action Q(state,action)
        #util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        self.Temporary_QValue = util.Counter()  #initializing a temporary QValue counter

        temporary_QValue = self.Temporary_QValue

        legal_Actions = self.getLegalActions(state)  #get all the legal actions like north,south,east,west,exit

        length_legalActions = len(legal_Actions)    #find length of legal actions just to find later if we have legal actions or not

        if length_legalActions == 0:  #to check if we have any legal action or not
            return 0.0    #Returns value 0 as we do not have any legal actions, we cannot pass 'None' as autograder in q8 expects a float value and not string value

        for a in legal_Actions:     #loop to check for each legal action

            temporary_QValue[a] = self.getQValue(state,a)     #Find the Qvalue of each action

        best_action = temporary_QValue.argMax() #find the best action to take in a state
        return best_action
        #util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        length_legalActions = len(legalActions)  #Find length of allowed actions

        if length_legalActions == 0:  #To check if no legal action is possible, that is incase of terminal state
              action = None     #set action as none and return from here
              return action

        epsilon = self.epsilon      #to get the epsilon value

        if util.flipCoin(epsilon):                     #util.flipcoin returns binary variable with probability p of success by using util.flipCoin(p), which returns True with probability p and False with probability 1-p.
              action =  random.choice(legalActions)     #Choosing randomly from list of allowed actions
              return action

        action = self.getPolicy(state) #Without probability epsilon we should take best policy action. getPolicy function calls the computeActionFromQValues function which gives us the best action to take in a state

        #util.raiseNotDefined()

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        Q_Value = self.Q #calling constructor

        learning_rate = self.alpha  #gives us the learning rate

        temporary_QValue = self.getQValue(state,action) #to get the Q value of the state

        nextState_QValue = self.getValue(nextState) #to get the Q value of the landing state when taken action a and state s

        discount_factor =  self.discount #to get the gamma/ discount factor


        Q_Value[(state,action)] = ((1-learning_rate) * temporary_QValue) + (learning_rate * (reward + discount_factor * nextState_QValue)) #for formula go to README_Reinforcement.txt at line 8

        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        Q_Value = 0 #initializing q value

        feat_Extractor = self.featExtractor

        weight = self.weights       #To get the weight to control exploration and exploitation

        features = feat_Extractor.getFeatures(state,action)  #to get all the features associated with (state,action) pair

        for each_feature in features:
             #refer to README_Reinforcement.txt for the formula at line 11
                  temp_Qvalue = weight[each_feature] * features[each_feature]   #Q(state,action) = w * featureVector where * is the dotProduct operator
                  Q_Value = Q_Value + temp_Qvalue

        return Q_Value   #Returns final qvalue
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state,action)

        learning_rate = self.alpha  #gives us the learning rate

        temporary_QValue = self.getQValue(state,action) #to get the Q value of the state,action pair

        nextState_QValue = self.getValue(nextState) #to get the Q value of the landing state when taken action a and state s

        discount_factor =  self.discount #to get the gamma/ discount factor

        weight = self.weights

        Q_Value = 0

        difference = (reward + discount_factor * nextState_QValue ) - (temporary_QValue)  #refer to README_Reinforcement.txt for the formula

        for each_feature in features:

            #refer to README_Reinforcement.txt for the formula at line 20
              weight[each_feature] = weight[each_feature] + learning_rate * difference * features[each_feature]

        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
