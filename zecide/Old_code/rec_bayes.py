import numpy as np
import pandas as pd
from pprint import pprint

'''
db = {'F': {'X': 0.54, 'factors': ['f1', 'f2', 'f3', 'f4', 'f5', 'f6'], 'parent': None},
    'f1': {'X': 0.78, 'Z': 0.24, 'factors': [], 'parent': 'F'},
    'f2': {'X': 0.4212, 'Z': 0.46, 'factors': [], 'parent': 'F'},
    'f3': {'X': 0.36, 'Z': 0.22, 'factors': [], 'parent': 'F'},
    'f4': {'X': 0.21, 'Z': 0.47, 'factors': [], 'parent': 'F'},
    'f5': {'X': 0.36, 'Z': 0.31, 'factors': [], 'parent': 'F'},
    'f6': {'X': 0.61, 'Z': 0.23, 'factors': [], 'parent': 'F'}}
'''
factor = ['f1', 'f2','f3','f4','f5','f6']
db = {'F': {'X': 0.54, 'factors': factor, 'parent': None},
    'f1': {'y': 0.80,  'parent': 'F'},
    'f2': {'y': 0.31,  'parent': 'F'},
    'f3': {'y': 0.22,  'parent': 'F'},
    'f4': {'y': 0.21,  'parent': 'F'},
    'f5': {'y': 0.36,  'parent': 'F'},
    'f6': {'y': 0.61,  'parent': 'F'}}

def bayes(x, y, z):

    return((x*y) / ((x*y) + z*(1-x)))

def update(factor,fac):
    if fac not in factor:
        factor.append(fac)

def calc(s):

    pprint(db[s])
    parent = db[s]['parent']
    factors = db[s]['factors']
    strt = db['f1']['y']
    for i in range(1,len(factors)):
        calc = bayes(db[s]['X'], strt, db['f'+str(i+1)]['y'])
        strt = calc
        

    if parent:
        calc(parent)
    return strt

if __name__ == "__main__":

    print(calc('F'))
    print("Result: ", db['F']['X'])
