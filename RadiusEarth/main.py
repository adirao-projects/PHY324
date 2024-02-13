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

WIRE_DISTANCE = 0.222 #m
SCOPE_DISTANCE = 2.082 - 0.225 #m
print(SCOPE_DISTANCE)
WIRE_LENGTH = 0.292 #m

AMMETER_UNCERT = 2 #2% + 0.5%

HUMAN_DIST_ERROR = 10 #%
HUMAN_SCOPE_ERROR = 0.1 #cm

HUMAN_SCOPE_ERROR_m = HUMAN_SCOPE_ERROR/100

GRAVITY = 9.81 # m/s

# Defining Units
meter = unit('m')
cm = unit('cm')

def save_data(data, title):
    with open(f'figures/{title}.txt', 'w') as f:
        print(data, file=f)
    

def load_data(sheet_name:str):
    df = pd.read_excel('data.xlsx', sheet_name=sheet_name)
    df['Delta D(cm)'] = df['D(cm)'] - df['D(cm)'].iloc[0]
    df['Delta D(cm) abs'] = df['Delta D(cm)'].abs()
    df['d(cm)'] = df['D(cm)']*(SCOPE_DISTANCE/WIRE_DISTANCE)
    return df

def convert_similar_triangles(side_length):
    ratio = SCOPE_DISTANCE/WIRE_DISTANCE  
    return side_length*ratio

def ammeter_uncertainty(current):
    if current<=5.1:
        return current*0.02 + 5.1000*0.005
    else:
        return current*0.02 + 21.000*0.005

def graph_data(df, weight, exp_num):
    #current = [ufloat(i, ammeter_uncertainty(i)) for i in df['I(A)']]
    #deflection= [ufloat(d, 0.05) for d in df['D(cm)']]
    
    current_sq = df['I(A)'].pow(2)
    #deflection = df['D(cm)'] - df['D(cm)'].iloc[0]
    deflection = df['Delta D(cm) abs']
    #pstd = np.diag(data['pcov'])
    uncert=[0.05]*len(current_sq) #[0.05]*len(current_sq)
    
    data = tk.curve_fit_data(current_sq, deflection, 'linear', 
                             uncertainty=uncert, chi=True, res=True)
    
    #save_data(data, f'Plot for {weight} g (Exp: {exp_num})')
    
    meta = {'title' : f'Plot for {weight} g (Exp: {exp_num})',
            'xlabel' : '$Current^2$ (A)',
            'ylabel' : 'Deflection (cm)',
            'chi_sq' : data['chi-sq']}
    
    tk.quick_plot_residuals(current_sq, deflection, data['graph-horz'], 
                            data['graph-vert'],
                            data['residuals'], meta, uncertainty=uncert)
                            #save=True)
    
    print(i,j, "GRADIENT:", data['popt'], "STANDARD DEV:", \
          np.diag(np.sqrt(data['pcov'])), r"CHI-SQ: %2.2f"%data['chi-sq'])
    #print(data)
    return (data['graph-horz'], data['graph-vert'], \
            data['popt'], np.sqrt(np.diag(data['pcov'])))


def graph_data_mass(df, weight, exp_num):
    #current = [ufloat(i, ammeter_uncertainty(i)) for i in df['I(A)']]
    #deflection= [ufloat(d, 0.05) for d in df['D(cm)']]
    
    mass = weight
    #deflection = df['D(cm)'] - df['D(cm)'].iloc[0]
    deflection = df['Delta D(cm) abs']
    current_sq = df['I(A)'].pow(2)
    #pstd = np.diag(data['pcov'])
    uncert=[0.05]*len(current_sq) #[0.05]*len(current_sq)
    
    data = tk.curve_fit_data(current_sq, deflection, 'inverse', 
                             uncertainty=uncert, chi=True, res=True)
    
    #save_data(data, f'Plot for {weight} g (Exp: {exp_num})')
    
    meta = {'title' : f'Plot for {weight} g (Exp: {exp_num})',
            'xlabel' : '$Current^2$ (A)',
            'ylabel' : 'Deflection (cm)',
            'chi_sq' : data['chi-sq']}
    
    tk.quick_plot_residuals(current_sq, deflection, data['graph-horz'], 
                            data['graph-vert'],
                            data['residuals'], meta, uncertainty=uncert)
                            #save=True)
    
    print(i,j, data['popt'])
    
    return (data['graph-horz'], data['graph-vert'], data['popt'][0])
    
    

def permitivity_gradient_equation(grad, mass):
    mu = (2*np.pi*mass*GRAVITY*grad)/WIRE_LENGTH
    
    return mu
    
    

def compare_curves(dic):
    plt.figure(figsize=(10,6))
    """
    for i in range(0,40, 10):
        for j in range(1,4):
            key = f'{i}-{j}'
            data = dic[key]
            x_data = data[0]
            y_data = data[1]
        
        plt.plot(x_data, y_data, label=key, linestyle='dashed')
    """  
    plt.title('Deflection Comparison')
    
    m = []
    
    x_data, y_data, m0, s0 = dic['0-1']
    
    m0 = ufloat(m0, s0)
    
    plt.plot(x_data, y_data, label='0', linestyle='dashed')
    print("0:", "m_0 TBD")
    
    x_data, y_data, m10, s10 = dic['10-3'] 
      
    plt.plot(x_data, y_data, label='10', linestyle='dashed')
    print("10:", "m_0", get_init_mass(m0, m10, 10))
    
    m.append(get_init_mass(m0, m10, 10))
    
    x_data, y_data, m20, s20 = dic['20-3']       
    plt.plot(x_data, y_data, label='20', linestyle='dashed')
    print("20:", "m_0", get_init_mass(m0, m20, 20))
    m.append(get_init_mass(m0, m20, 20))
    
    x_data, y_data, m30, s30 = dic['30-2']       
    plt.plot(x_data, y_data, label='30', linestyle='dashed')
    print("30:", "m_0", get_init_mass(m0, m30, 30))
    m.append(get_init_mass(m0, m30, 30))
    
    x_data, y_data, m40, s40 = dic['40-1']       
    plt.plot(x_data, y_data, label='40', linestyle='dashed')
    print("40:", "m_0", get_init_mass(m0, m40, 40))
    m.append(get_init_mass(m0, m40, 40))
    
    x_data, y_data, m50, s50 = dic['50-3']       
    plt.plot(x_data, y_data, label='50', linestyle='dashed')
    print("50:", "m_0", get_init_mass(m0, m50, 50))
    m.append(get_init_mass(m0, m50, 50))
    
    print("AVG M: ", sum(m)/len(m), )
    
    #x_data, y_data, m = dic['500-1']       
    #plt.plot(x_data, y_data, label='500-1', linestyle='dashed')
    #x_data, y_data, m = dic['100-1']
    #plt.plot(x_data, y_data, label='100-1', linestyle='dashed')
    plt.xlabel('$Current^2$ (A)')
    plt.ylabel('Deflection (cm)')
    
    plt.legend(loc='upper left', title='$m_\Delta \ (mg)$', fontsize='xx-small', 
               title_fontsize='x-small')
    
    plt.savefig('figures/compare.png')
    

def get_init_mass(grad_1, grad_2, mass_2):
    print(grad_1, grad_2, mass_2)
    grad_ratio = grad_2/grad_1
    m_0 = grad_ratio*(mass_2/(1-grad_ratio))
     
    return m_0
     

if __name__ == '__main__':
    
    exp_plots = {}
    
    for i in range(0,60, 10):
        for j in range(1,4):
            df = load_data(f'{i}-{j}')
            best_fit = graph_data(df, i, j)
            exp_plots[f'{i}-{j}'] = best_fit
    
    df = load_data('500-1')
    exp_plots['500-1'] = graph_data(df, 500, 1)
    
    df = load_data('100-1')
    exp_plots['100-1'] = graph_data(df, 500, 1)
    
    grad_m0 = exp_plots['0-1'][2][0]
    grad_m500 = exp_plots['500-1'][2][0]
    
    m_0 = get_init_mass(grad_m0, grad_m500, 500) #mg
    
    mu_0 = permitivity_gradient_equation(grad_m0, m_0/1000000)
    print('Mu_0={:e}'.format(mu_0))
    
    #exp_plots['100-1'] = 
    compare_curves(exp_plots)
    
    #df0_1 = load_data('0-1')
    #df0_2 = load_data('0-2')
    #df0_3 = load_data('0-3')
    
    #graph_data(df0_1)
    #graph_data(df0_2)
    #graph_data(df0_3)
    