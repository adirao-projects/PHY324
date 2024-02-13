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

import toolkit as tk

HUMAN_DIST_ERROR = 10 #%
HUMAN_SCOPE_ERROR = 0.1 #cm

HUMAN_SCOPE_ERROR_m = HUMAN_SCOPE_ERROR/100


FLOOR_HEIGHT = 3.95 # m
COUNTER_CONST = 0.1005 # Miligals

GRAV_CONST = 6.67e-11
SUN_MASS = 2.0e30

GRAVITY = 9.804253 # m/s

# Defining Units
meter = unit('m')
cm = unit('cm')

def save_data(data, title):
    with open(f'figures/{title}.txt', 'w') as f:
        print(data, file=f)
    

def load_data(sheet_name:str):
    df = pd.read_excel('data.xlsx', sheet_name=sheet_name)
    df['Counter(mGal)'] = df['Counter'].mul(COUNTER_CONST)
    df['Height(m)'] = df['Floor'].mul(FLOOR_HEIGHT)
    
    return df


def graph_data(df, expnum):
    height = df['Height(m)']
    counter = df['Counter(mGal)']
    uncert = [4*COUNTER_CONST]*len(height)
    
    #print(height, counter, uncert)
    data = tk.curve_fit_data(height, counter, 'linear-int', 
                             uncertainty=uncert, chi=True, res=True)
    
    #save_data(data, f'Plot for {weight} g (Exp: {exp_num})')
    
    meta = {'title' : f'Floor-mGal plot',
            'xlabel' : 'Floor Height (m)',
            'ylabel' : 'Counter (mGal)',
            'chi_sq' : data['chi-sq']}
    
    tk.quick_plot_residuals(height, counter, data['graph-horz'], 
                            data['graph-vert'],
                            data['residuals'], meta, uncertainty=uncert)
                            #save=True)
    
    print(meta['chi_sq'])
    
    return (data['graph-horz'], data['graph-vert'], \
            data['popt'], np.sqrt(np.diag(data['pcov'])))


if __name__ == '__main__':
    
    exp_plots = {}
    df = load_data('Exp1')
    graph_data(df, 1)
    
    df = load_data('Exp2')
    graph_data(df, 2)
    
    