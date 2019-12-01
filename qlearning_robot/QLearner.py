"""
Template for implementing QLearner  (c) 2015 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---
Student Name: Akshay Karthik
GT User ID: akarthik3
GT ID: 903212846
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, num_states = 100, num_actions = 4, alpha = 0.2, gamma = 0.9, rar = 0.5, radr = 0.99, dyna = 0, verbose = False):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        self.s = 0
        self.a = 0

        self.experiences = []
        self.QTable = np.zeros(shape = (num_states, num_actions))

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        a = rand.randint(0, self.num_actions - 1) if rand.random() <= self.rar else np.argmax(self.QTable[s, :])
        self.s, self.a = s, a
        if self.verbose: print("s = ", s, ", a = ", a)
        return a

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The reward
        @returns: The selected action
        """

        self.QTable[self.s, self.a] = (1 - self.alpha) * self.QTable[self.s, self.a] + self.alpha * (self.gamma * self.QTable[s_prime, np.argmax(self.QTable[s_prime, :])] + r)
        self.experiences.append((self.s, self.a, s_prime, r))

        for index in range(self.dyna):
            instance = rand.randint(0, len(self.experiences) - 1)
            dynaS, dynaA, dynaS_prime, dynaR = self.experiences[instance][0], self.experiences[instance][1], self.experiences[instance][2], self.experiences[instance][3]
            self.QTable[dynaS, dynaA] = (1 - self.alpha) * self.QTable[dynaS, dynaA] + self.alpha * (self.gamma * self.QTable[dynaS_prime, np.argmax(self.QTable[dynaS_prime, :])] + dynaR)

        self.rar *= self.radr
        a = self.querysetstate(s_prime)
        if self.verbose: print("s = ", s_prime, ", a = ", a, ", r = ", r)
        return a

    def author(self):
        return 'akarthik3'

if __name__=="__main__":
    print("Remember Q from Star Trek? Well, this isn't him")