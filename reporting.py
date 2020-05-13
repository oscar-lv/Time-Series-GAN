#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 17:41:53 2020

@author: oscar
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_gan_training(d_loss,g_loss):
    return 

def plot_dist(prices : pd.DataFrame, title:str='Distribution Plot',save:bool=False, fname:str='distplot'):
    lrets = np.log(prices) - np.log(prices.shift(1))
    sns.distplot(lrets, bins=30, color='skyblue', axlabel='Log Returns', label='Log returns')
    plt.title(title)
    if save==True:
        plt.savefig(fname, dpi=500)
    return None

def plot_gan_history(history, save=False):
    plt.plot(history[0], label='discriminator')
    plt.plot(history[2],label='generator')
    plt.legend(['Discriminator Loss', 'Generator Loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if save==True:
        plt.savefig('traininghistory', dpi=500)
    plt.show()