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
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            nextValues = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    nextValues[state] = 0.0
                else:
                    maxValue = float('-inf')
                    for action in self.mdp.getPossibleActions(state):
                        value = self.computeQValueFromValues(state, action)
                        if (value > maxValue):
                            maxValue = value
                            nextValues[state] = maxValue

            self.values = nextValues


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

        returnValue = 0.0

        for transState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            returnValue += (prob * (self.mdp.getReward(state, action, transState) + (self.discount * self.values[transState])))

        return returnValue
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        if self.mdp.isTerminal(state):
            return None
        else:
            value = float('-inf')
            policy = None
            for action in self.mdp.getPossibleActions(state):
                qValue = self.computeQValueFromValues(state, action)
                if (value < qValue):
                    value = qValue
                    policy = action
            return policy


        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # for i in range(self.iterations):
        #     nextValues = util.Counter()
        #     for state in self.mdp.getStates():
        #         if self.mdp.isTerminal(state):
        #             nextValues[state] = 0.0
        #         else:
        #             maxValue = float('-inf')
        #             for action in self.mdp.getPossibleActions(state):
        #                 value = self.computeQValueFromValues(state, action)
        #                 if (value > maxValue):
        #                     maxValue = value
        #                     nextValues[state] = maxValue
        #
        #     self.values = nextValues

        predecessors = []

        pq = util.PriorityQueue()

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                self.values[state] = 0.0
            else:
                maxValue1 = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    value = self.computeQValueFromValues(state, action)
                    if (value > maxValue1):
                        maxValue1 = value
                diff = abs(self.values[state] - maxValue1)
                pq.push(state, -diff)
        for i in range(self.iterations):
            if pq.isEmpty():
                break
            else:
                newState = pq.pop()
                if self.mdp.isTerminal(newState):
                    self.values[newState] = 0.0
                else:
                    maxValue2 = float('-inf')
                    for action in self.mdp.getPossibleActions(newState):
                        value = self.computeQValueFromValues(newState, action)
                        if (value > maxValue2):
                            maxValue2 = value
                    self.values[newState] = maxValue2
                predecessorSet = self.findPredecessors(newState)
                for predecessor in predecessorSet:
                    maxValue3 = float('-inf')
                    for action in self.mdp.getPossibleActions(predecessor):
                        value = self.computeQValueFromValues(predecessor, action)
                        if (value > maxValue3):
                            maxValue3 = value
                    diff = abs(self.values[predecessor] - maxValue3)
                    pq.push(predecessor, -diff)

    def findPredecessors(self, state):

        predecessors = set()
        for s in self.mdp.getStates():
            for a in self.mdp.getPossibleActions(s):
                successors = self.mdp.getTransitionStatesAndProbs(s, a)
                if (s in successors) and (successors[s][1] > 0):
                    predecessors.add(s)
        return predecessors
