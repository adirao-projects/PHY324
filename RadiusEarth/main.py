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

import toolkit as tk

HUMAN_DIST_ERROR = 10 #%
HUMAN_SCOPE_ERROR = 0.1 #cm

HUMAN_SCOPE_ERROR_m = HUMAN_SCOPE_ERROR/100

STEP = ufloat(0.177, 0.0005)

FLOOR_HEIGHT_STEP = STEP*22

FLOOR_HEIGHT = 3.95 # m
COUNTER_CONST = 0.1005 # Miligals

GRAV_CONST = 6.67e-11
SUN_MASS = 2.0e30

GRAV_ACCEL = 9.804253 # m/s

# Defining Units
meter = unit('m')
cm = unit('cm')

def save_data(data, title):
    with open(f'figures/{title}.txt', 'w') as f:
        print(data, file=f)
    

def load_data(sheet_name:str):
    df = pd.read_excel('data.xlsx', sheet_name=sheet_name)
    df['Counter(mGal)'] = df['Counter'].mul(COUNTER_CONST)
    df['Height(m)'] = df['Floor'].mul(FLOOR_HEIGHT_STEP.n)
    
    return df


def graph_data(df, expnum):
    #height = ufloat(df['Height(m)'], FLOOR_HEIGHT_STEP.s)
    height = df['Height(m)']
    counter = df['Counter(mGal)']
    uncert = [4*COUNTER_CONST]*len(height)

    uncert_x = [4*int(FLOOR_HEIGHT_STEP.s)]*len(height)
    
    #print(height, counter, uncert)
    data = tk.curve_fit_data(height, counter, 'linear-int', 
                             uncertainty=uncert, chi=True, res=True,
                             uncertainty_x = uncert_x)
    
    #save_data(data, f'Plot for {weight} g (Exp: {exp_num})')
    
    meta = {'title' : f'Floor-mGal plot',
            'xlabel' : 'Floor Height (m)',
            'ylabel' : 'Counter (mGal)',
            'chi_sq' : data['chi-sq'],
            'data-label': 'data point',
            'fit-label': '$g = \zeta h + g_0$',
            'save-name': 'exp1_data.png',
            'loc': 'lower left'}
    
    tk.quick_plot_residuals(height, counter, data['graph-horz'], 
                            data['graph-vert'],
                            data['residuals'], meta, uncertainty=uncert, 
                            uncertainty_x = uncert_x)
                            #save=True)
    
    print(meta['chi_sq'])
    
    data['pstd'] = np.sqrt(np.diag(data['pcov']))
    
    return data


def calculate_radius_earth(gradient, grav = GRAV_ACCEL):
    return (-2)*(grav)*(1/gradient)


def convert_mGal(quantity):
    return quantity*0.00001
    

if __name__ == '__main__':
    print(FLOOR_HEIGHT_STEP, FLOOR_HEIGHT)
    
    for i in range(1, 5):
        df = load_data(f'Exp{i}')
        data = graph_data(df, 1)
        #print(data['popt'])
        grad, grav0 = data['popt']
        grad = ufloat(grad, data['pstd'][0])
        grad = convert_mGal(grad)
        grav0 = convert_mGal(grav0)
        
        grav0 = ufloat(grav0, data['pstd'][1])
        
        R = calculate_radius_earth(grad)
        
        print(f'Radius of Earth {R}')
        

    