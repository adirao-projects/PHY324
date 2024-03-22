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
    df = pd.read_csv(f"{path}exp{exp_num}.txt", sep="\t", skiprows=[0,1],
                     names=['Pos(Rad)', 'Intensity(Volts)'])
    df['Pos(Rad)'].astype(float)
    df['Intensity(Volts)'].astype(float)
    
    return df
    
def brewster(exp_num, polar_type):
    if polar_type == 'H':
        polar_type = "Horizontal"
        
    elif polar_type == 'V':
        polar_type = "Vertical"
        
    elif polar_type == 'N':
        polar_type = "No"
    
    df = load_data("data/Brewster/brew", exp_num)
    #print(xdata)
    #print(ydata)
    #df.plot.scatter(x='Pos(Rad)', y='Intensity(Volts)', c='red', s=0.2, 
    #                grid=True)
    
    df['pct'] = df['Intensity(Volts)'].pct_change()
    #print(df['pct'])
    
    #df['Pos(Rad)'] = df['Pos(Rad)'].div(180).multiply(np.pi)
    
    df = df[df['pct']<=0.03]
    df = df[df['pct']>=-0.03]
    df = df[df['Pos(Rad)']>=200]
    df = df[df['Intensity(Volts)']>=0.10]
    
    df['Pos(Rad)'] = df['Pos(Rad)'].div(180).multiply(np.pi)
    
    #print(df)
    
    df.plot.scatter(x='Pos(Rad)', y='Intensity(Volts)', c='red', s=0.2, 
                    grid=True, title=f"{exp_num}")
    #tk.quick_plot_test(xdata, ydata)

    uncert = [0.1]*len(df)
    
    data = tk.curve_fit_data(df['Pos(Rad)'].to_numpy(), df['Intensity(Volts)'].to_numpy(), 
                      fit_type='custom', model_function_custom=brewster_eqn,
                      uncertainty=uncert, res=True, chi=True, guess=(1.49, 0.15, 2.5))
    
    meta = {'title' : f"Brewster's Law for {polar_type} Polarizer (Experiment {exp_num})",
            'xlabel' : r'Second Polarizer Angle ($\theta$) (rad)',
            'ylabel' : 'Recorded Intensity (Volts)',
            'chi_sq' : data['chi-sq'],
            'fit-label': r"$r_{\perp} = \frac{n_1\cos\theta_1 - n_2\cos\theta_2}{n_1\cos\theta_1 + n_2\cos\theta_2}$",
            'data-label': "Raw data",
            'save-name' : f'm2_{exp_num}',
            'loc' : 'lower right'}
    
    tk.quick_plot_residuals(df['Pos(Rad)'], df['Intensity(Volts)'], 
                           data['graph-horz'], data['graph-vert'],
                           data['residuals'], meta=meta, uncertainty=uncert, save=False)
    #x = np.linspace(3.5, 6, 100)
    #plt.plot(x, brewster_eqn(x, 1.49, 0.3, 2))
    
    print(meta['title'], 'CHI:', data['chi-sq'], 'PARAMS:', data['popt'], data['pstd'])
    
def deg_rad(angle):
    return (angle/180)*(np.pi)
    
def rad_deg(angle):
    return (angle/np.pi)*(180)

def malus2(exp_num):
    df = load_data("data/Malus/mtwo", exp_num)
    
    df.plot.scatter(x='Pos(Rad)', y='Intensity(Volts)', c='red', s=0.2, 
                    grid=True, title=f"{exp_num}")
    
    uncert = [0.1]*len(df)
    
    
    
    data = tk.curve_fit_data(df['Pos(Rad)'], df['Intensity(Volts)'], 
                      fit_type='custom', model_function_custom=malus2_eqn,
                      uncertainty=uncert, res=True, chi=True)
    
    meta = {'title' : f"Malus' Law for Two Polarizers (Experiment {exp_num})",
            'xlabel' : r'Second Polarizer Angle ($\theta$) (rad)',
            'ylabel' : 'Recorded Intensity (Volts)',
            'chi_sq' : data['chi-sq'],
            'fit-label': r"$I' = I_0 \cos^2 \theta$",
            'data-label': "Raw data",
            'save-name' : f'm2_{exp_num}',
            'loc' : 'lower right'}
    
    tk.quick_plot_residuals(df['Pos(Rad)'], df['Intensity(Volts)'], 
                           data['graph-horz'], data['graph-vert'],
                           data['residuals'], meta=meta, uncertainty=uncert, save=False)
    
    print(meta['title'],'CHI', data['chi-sq'], 'PARAMS', data['popt'], data['pstd'])
    
def malus3(exp_num):
    df = load_data("data/Malus/mthree", exp_num)
    
    df.plot.scatter(x='Pos(Rad)', y='Intensity(Volts)', c='red', s=0.2, 
                    grid=True, title=f"{exp_num}")
    
    uncert = [0.1]*len(df)
    
    data = tk.curve_fit_data(df['Pos(Rad)'], df['Intensity(Volts)'], 
                      fit_type='custom', model_function_custom=malus3_eqn,
                      uncertainty=uncert, res=True, chi=True)
    
    meta = {'title' : f"Malus' Law for Three Polarizers (Experiment {exp_num})",
            'xlabel' : r'Second Polarizer Angle ($\theta$) (rad)',
            'ylabel' : 'Recorded Intensity (Volts)',
            'chi_sq' : data['chi-sq'],
            'fit-label': r"$I' = I_0 \cos^2 \theta$",
            'data-label': "Raw data",
            'save-name' : f'm2_{exp_num}',
            'loc' : 'lower right'}
    
    tk.quick_plot_residuals(df['Pos(Rad)'], df['Intensity(Volts)'], 
                           data['graph-horz'], data['graph-vert'],
                           data['residuals'], meta=meta, uncertainty=uncert, save=False)
    
    print(meta['title'], 'CHI:', data['chi-sq'], 'PARAMS:', data['popt'], data['pstd'])
    

def malus2_eqn(angle, I0, offset):
    return I0*((np.cos(angle+offset))**2)

def malus2_eqn_imp(I0, angle, r1, r2, r3):
    I1 = (I0/(r1**2))
    I2 = I1*((np.cos(np.pi/2))**2)
    I3 = (I2/(r1**2))
    I4 = I3*((np.cos(angle))**2)
    I5 = I4*(I4/(r3**2))
    
    return I5


def malus3_eqn(angle, I0, offset):
    return (1/4)*(I0)*((np.sin(2*(angle + offset)))**2)


def brewster_eqn(theta1, n2, offset, y0):
    polar = np.pi/4
    n1 = 1
    theta1 = theta1 + offset
    theta2 = np.arcsin((n1/n2)*np.sin(theta1))
    #r_perp = (((n1*np.cos(theta1) - n2*np.cos(theta2))/(n1*np.cos(theta1) + n2*np.cos(theta2)))*(np.cos(polar)**2)) + y0
    #r_prll = (n1*np.cos(theta2) - n2*np.cos(theta1))/(n1*np.cos(theta2) + n2*np.cos(theta1))
    coeff = (np.cos(polar))**2
    r_perp = (n1*np.cos(theta1) - n2*np.cos(theta2))/(n1*np.cos(theta1) + n2*np.cos(theta2))
    r_perp = r_perp + y0
    r_perp = r_perp**2
    r_perp = 0.7*coeff*r_perp
    
    return r_perp**2


if __name__ == '__main__':
    #brewster_best = [1, 3, 6, 8, 9]
    brewster_best = [1,3, 6, 8, 9]
    for i in brewster_best:
        brewster(i, 'H')

        
    for i in range(1,4):
        malus2(i)
        
    for i in range(1,4):
        malus3(i)

        

    