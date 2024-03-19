# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:34:34 2024

@author: Aditya Rao 1008307761
"""

import numpy as np
from units import unit
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from uncertainties import ufloat
import datetime as dt
import toolkit as tk
    

def load_data(path, exp_num):
    df = pd.read_csv(f"{path}exp{exp_num}.txt", sep="\t", skiprows=[0])
    return df
    
def brewster(exp_num):
    df = load_data("data/Brewster/brew", exp_num)
    print(df)

def malus2(exp_num):
    df = load_data("data/Malus/mtwo", exp_num)
    print(df)
    
    
def malus3(exp_num):
    df = load_data("data/Malus/mthree", exp_num)
    print(df)

if __name__ == '__main__':
    brewster(1)
    malus2(1)
    malus3(1)
    
        

    