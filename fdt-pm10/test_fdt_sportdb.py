# ~~~~~~~~~~~~~~~~~
# test-fdt-sportsdb
# ~~~~~~~~~~~~~~~~~

# Testing/Debugging script for the FuzzyTree class
# The dataset is an example from Yuan - Induction of FDTs

import copy
import numpy as np
from fdt import FuzzyTree
from fuzzyset import FuzzySet
from fuzzifier import Fuzzifier
from fuzzyplotter import FuzzyPlotter
from fuzzyvariable import FuzzyVariable

# Universe of discourse
x = np.arange(1,17)

# Outlook variable  
outlook = FuzzyVariable('Outlook', x, terms=['Sunny', 'Cloudy', 'Rain'])
m_outlook = np.array([[0.9, 0.8, 0, 0.2, 0, 0, 0, 0, 1, 0.9, 0.7, 0.2, 0.9, 0, 0, 1],
             [0.1, 0.2, 0.7, 0.7, 0.1, 0.7, 0.3, 1, 0, 0.1, 0.3, 0.6, 0.1, 0.9, 0, 0],
             [0, 0, 0.3, 0.1, 0.9, 0.3, 0.7, 0, 0, 0, 0, 0.2, 0, 0.1, 1, 0]])
outlook.setmval(m_outlook.T)

# Temperature variable
temperature = FuzzyVariable('Temperature', x, terms=['Hot', 'Mild', 'Cool'])
m_temp = np.array([[1, 0.6, 0.8, 0.3, 0.7, 0, 0, 0, 1, 0, 1, 0, 0.2, 0, 0, 0.5],
          [0, 0.4, 0.2, 0.7, 0.3, 0.3, 0, 0.2, 0, 0.3, 0, 1, 0.8, 0.9, 0, 0.5],
          [0, 0, 0, 0, 0, 0.7, 1, 0.8, 0, 0.7, 0, 0, 0, 0.1, 1, 0]])
temperature.setmval(m_temp.T)

# Humidity variable
humidity = FuzzyVariable('Humidity', x, terms=['Humid', 'Normal'])
m_hum= np.array([[0.8, 0, 0.1, 0.2, 0.5, 0.7, 0, 0.2, 0.6, 0, 1, 0.3, 0.1, 0.1, 1, 0],
        [0.2, 1, 0.9, 0.8, 0.5, 0.3, 1, 0.8, 0.4, 1, 0, 0.7, 0.9, 0.9, 0 ,1]])
humidity.setmval(m_hum.T)

# Wind variable
wind = FuzzyVariable('Wind', x, terms=['Windy', 'Not_windy'])
m_wind= np.array([[0.4, 0, 0.2, 0.3, 0.5, 0.4, 0.1, 0, 0.7, 0.9, 0.2, 0.3, 1, 0.7, 0.8, 0],
         [0.6, 1, 0.8, 0.7, 0.5, 0.6, 0.9, 1, 0.3, 0.1, 0.8, 0.7, 0, 0.3, 0.2, 1]])
wind.setmval(m_wind.T)

# Plan variable
plan = FuzzyVariable('Plan', x, terms=['Volleyball', 'Swimming', 'Weight_lifting'])
m_plan= np.array([[0, 1, 0.3, 0.9, 0, 0.2, 0, 0.7, 0.2, 0, 0.4, 0.7, 0, 0, 0, 0.8],
         [0.8, 0.7, 0.6, 0.1, 0, 0, 0, 0, 0.8, 0.3, 0.7, 0.2, 0, 0, 0, 0.6],
         [0.2, 0, 0.1, 0, 1, 0.8, 1, 0.3, 0, 0.7, 0, 0.1, 1, 1, 1, 0]])
plan.setmval(m_plan.T)

# Build the fuzzy dataset
db = {0: copy.deepcopy(m_outlook.T),
      1: copy.deepcopy(m_temp.T),
      2: copy.deepcopy(m_hum.T),
      3: copy.deepcopy(m_wind.T),
      -1: copy.deepcopy(m_plan.T)}

db_test = [[db[i][j] for i in range(4)] for j in range(16)]

fvar = {0: outlook,
        1: temperature,
        2: humidity,
        3: wind,
        -1: plan}

# Build a FuzzyTree
alpha = 0.5
fdt = FuzzyTree(db, fvar, acut=alpha, tlvl=0.75)
fdt.build_tree()
fdt.print_tree()
fdt.print_rules()
r = fdt.predict(db_test)
c = fdt.to_single_class(r)
