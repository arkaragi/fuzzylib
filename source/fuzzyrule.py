# ~~~~~~~~~~~~
# fuzzyrule.py
# ~~~~~~~~~~~~

import numpy as np
from fuzzyset import FuzzySet
from fuzzyvariable import FuzzyVariable
from fuzzyplotter import FuzzyPlotter

class FuzzyRule(object):

    def __init__(self, expression):
        self.antecedent = expression[0]
        self.consequent = expression[-1]

    def __repr__(self):
        return 'FuzzyRule\nAntecedent: {}\nConcequent: {}' \
               .format(self.antecedent, self.consequent)

    def evaluate_antecedent(self, values):
        m = []
        for fuzz, v in zip(self.antecedent, values):
            #m.append(fuzz(v))
            m.append(fuzz.func.trimf(np.array([v]), f.prmts))
        act_deg = []
        if len(m) == 1:
            act_deg.append(m[0])
        else:
            for i, t in enumerate(self.antecedent_ops):
                if t is '&': act_deg.append(min(m))
                if t is '|': act_deg.append(max(m))
        return act_deg

class FuzzyControlSystem(object):

    def __init__(self, rules):
        self.rules = rules
        self.consequents = [r.consequent for r in rules]
        self.inputs = []

    def activate(self):
        self.act_degree = []
        for rule in self.rules:
            a = rule.evaluate_antecedent(self.inputs)
            self.act_degree.extend(a)

    def evaluate_consequent(self):
        for i, con in enumerate(self.consequents):
            p = float(self.act_degree[i])
            con.cutpoint(p)

    def defuzz(self, method=None):
        final = FuzzySet.Union(self.consequents)
        temp = list()
        print(max(final.m))
        for i in range(len(final.x)):
            if final.m[i] >= max(final.m):
                temp.append(final.x[i])
        val = sum(temp)/len(temp)
        fplot = FuzzyPlotter([final])
        fplot()
        return val

