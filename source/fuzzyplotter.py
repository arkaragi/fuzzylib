# fuzzyplotter.py

import numpy as np
import matplotlib.pyplot as plt
from fuzzyset import FuzzySet
from fuzzyvariable import FuzzyVariable
from fuzzifier import FuzzyDataset

class FuzzyPlotter(object):

    def __init__(self, fuzzy_obj, xlabel=None, ylabel=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        if isinstance(fuzzy_obj, list):
            self.fuzzy_obj = fuzzy_obj
            self._plot_list()
        elif isinstance(fuzzy_obj, FuzzyVariable):
            self.fuzzy_obj = fuzzy_obj
            self._plot_fuzzyvar()
        elif isinstance(fuzzy_obj, FuzzySet):
            self.fuzzy_obj = fuzzy_obj
            self._plot_fuzzyset()
        elif isinstance(fuzzy_obj, FuzzyDataset):
            self.fuzzy_obj = fuzzy_obj
            self._plot_fuzzydata()
        
##    def __call__(self):
##        plt.figure()
##        plt.title('Fuzzy Set')
##        for fuzz in self.fuzzy_obj:
##            plt.plot(fuzz.x, fuzz.m)
##            plt.xlabel('Variable')
##            plt.ylabel('Membership Degree')
##        plt.show()

    def _plot_list(self):
        plt.figure()
        plt.title('Fuzzy Set')
        for i, fuzz in enumerate(self.fuzzy_obj):
            plt.plot(fuzz.x, fuzz.m)
        plt.xlim(fuzz.x[0] - 5e-1, fuzz.x[-1] + 5e-1)
        plt.ylim(0, 1.2)
        plt.xlabel('X')
        plt.ylabel('μ(x)')
        plt.show()

    def _plot_fuzzyset(self, obj=None):
        plt.figure()
        plt.title('Fuzzy Set')
        plt.plot(self.fuzzy_obj.x, self.fuzzy_obj.m)
        plt.xlim(self.fuzzy_obj.x[0] - 5e-1, self.fuzzy_obj.x[-1] + 5e-1)
        plt.ylim(0, 1.2)
        plt.xlabel('X')
        plt.ylabel('μ(x)')
        plt.show()
    
    def _plot_fuzzyvar(self, obj=None):
        x = self.fuzzy_obj.x
        name = self.fuzzy_obj.name
        fuzz = self.fuzzy_obj.fuzzy
        terms = self.fuzzy_obj.terms
        fig = plt.figure()
        plt.title('{}'.format(name))
        for i, f in enumerate(fuzz):
            plt.plot(x, f.func(x), label=terms[i])
        if self.xlabel is None:
            plt.xlabel('Universe of Discourse')
        else:
            plt.xlabel(self.xlabel)
        if self.ylabel is None:
            plt.ylabel('Membership Degree')
        else:
            plt.ylabel(self.ylabel)
        plt.legend()
        plt.show()

    def _plot_fuzzydata(self):
        for i, fvar in enumerate(self.fuzzy_obj.fuzzy_variables.values()):
            plt.figure()
            plt.title('{}'.format(fvar.name))
            x = fvar.x
            terms = fvar.terms
            for j, fuzz in enumerate(fvar.fuzzy):
                s = 'fuzz.func.{}(x, fuzz.prmts)'.format(fuzz.mfunc)
                plt.plot(x, eval(s), label=terms[j])
            plt.xlim(x[0],x[-1])
            plt.ylim(0, 1.2)
            plt.xlabel('{}'.format(fvar.name))
            plt.ylabel('μ(x)')
            plt.legend()
            #plt.show()
            plt.savefig('{}'.format(fvar.name))
        
##        n = len(self.fuzzy_obj)-1
##        fig, axes = plt.subplots(n//2+1, 2, figsize=(16, 16), sharey=True)
##        for i, ax in enumerate(axes.flatten()):
##            try:
##                fvar = self.fuzzy_obj.fuzzy_variables[i]
##                x = fvar.x
##                name = fvar.name
##                fuzz = fvar.fuzzy
##                terms = fvar.terms
##                ax.set_title('{}'.format(name), fontsize=9)
##                for i, f in enumerate(fuzz):
##                    s = 'f.func.{}(x, f.prmts)'.format(f.mfunc)
##                    ax.plot(x, eval(s), label=terms[i])
##                ax.legend(fontsize=6)
##                ax.set_ylabel('μ(x)')
##            except KeyError:
##                fvar = self.fuzzy_obj.fuzzy_variables[-1]
##                x = fvar.x
##                name = fvar.name
##                fuzz = fvar.fuzzy
##                terms = fvar.terms
##                ax.set_title('{}'.format(name), fontsize=9)
##                for i, f in enumerate(fuzz):
##                    s = 'f.func.{}(x, f.prmts)'.format(f.mfunc)
##                    ax.plot(x, eval(s), label=terms[i])
##                ax.legend(fontsize=6)
##                ax.set_ylabel('μ(x)')
##            plt.show()
##        plt.subplots_adjust(left=0.15, wspace=0.2, hspace=0.4)    
##        plt.show()


### Testing/Debbuging 
##uod = np.arange(-10,50,0.1)
##A = FuzzySet(uod)
##A.set_mf('trimf', [-5,2,12])
##B = FuzzySet(uod)
##B.set_mf('trapmf', [8,14,22,28])
##C = FuzzySet(uod)
##C.set_mf('gaussmf', [30,4])
##t = ['Low', 'Average', 'High']
##fv = FuzzyVariable('Temperature', uod, t)
##fv.setfuzzy([A,B,C])
##plot = FuzzyPlotter(fv)

