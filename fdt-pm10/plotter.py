# ~~~~~~~~~~~~~~~~
# plotter.py
# ~~~~~~~~~~~~~~~~
        
import numpy as np
from fuzzyset import FuzzySet
from fuzzyplotter import FuzzyPlotter
from fuzzyvariable import FuzzyVariable
from matplotlib import pyplot as plt

### pic temperature
##name = 'Temperature'
##uod = np.arange(-10, 50, 0.01)
##terms = ['Very Low', 'Low', 'Average', 'High', 'Very High']
##v = FuzzyVariable(name, uod, terms)
##v.setmf([('trimf', [0, 0, 10]),
##         ('trimf', [5, 12, 20]),
##         ('trimf', [15, 20, 25]),
##         ('trimf', [20, 28, 35]),
##         ('trimf', [30, 40, 40])])
##FuzzyPlotter(v)

# pic1
name = 'Classic vs Fuzzy Logic: Describing the class of tall humans'

uod = np.arange(100, 250, 0.01)
terms = ['Characteristic function', 'Membership function']
v = FuzzyVariable(name, uod, terms)
v.setmf([('trapmf', [180,180,250,250]), ('trapmf', [150,180,250,250])])
FuzzyPlotter(v)

### pic1
##name = 'Parametric Membership Functions'
##uod = np.arange(-10, 50, 0.01)
##terms = ['trimf, [-5,2,12]', 'trapmf, [8,14,22,28]', 'gaussmf, [30,4]']
##v = FuzzyVariable(name, uod, terms)
##v.setmf([('trimf', [-5,2,12]),
##         ('trapmf', [8,14,22,28]),
##         ('gaussmf', [30,4])])
##FuzzyPlotter(v)

##### pic 2345
##uod = np.arange(-10,110,0.1)
##A = FuzzySet(uod)
##A.set_mf('trimf', [0,35,60])
##B = FuzzySet(uod)
##B.set_mf('trimf', [20,75,95])
##C = A | B
##D = A & B
##E = ~A
##F = ~B
##plt.figure()
##plt.plot(A.x,A.m,'r')
##plt.plot(E.x,E.m,'g')
##plt.title('Complementary Operation')
###plt.title('The intersection of A and B')
##plt.xlabel('Universe of Discourse')
##plt.ylabel('Membership Degree')
##plt.legend(['μA', 'μAcomplement'])
##plt.show()

##flist = [A, B, C, D]
##tlist = ['Fuzzy set A', 'Fuzzy set B', 'Union', 'Intersection']
##colors = ['r', 'g']
##fig, axes = plt.subplots(2,1,figsize=(8,8),sharey=True)
##for i, ax in enumerate(axes.flatten()):
##    if i == 0:
##        for f in flist[:i+2]:
##            ax.plot(f.x, f.m)
##            ax.set_title('Fuzzy sets A and B')
##            ax.set_xlabel('Universe of Discourse')
##            ax.set_ylabel('Membership Degree')
##        ax.legend(['μA', 'μB'])
##    else:
##        for j, f in enumerate(flist[i+1:]):
##            ax.plot(f.x, f.m, colors[j])
##            ax.set_title('Union and Intersection')
##            ax.set_xlabel('Universe of Discourse')
##            ax.set_ylabel('Membership Degree')
##        ax.legend(['Union', 'Intersection'])
##plt.subplots_adjust(hspace=0.4)
##plt.show()

