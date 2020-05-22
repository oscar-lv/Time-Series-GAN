#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:53:04 2020

@author: oscar
"""

from reporting import plot_gan_history, plot_dist, generate_samples, plot_gan_result
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import numpy as np
from data_utils import generate_sp_samples
from scipy.stats import entropy

"""
Plotting Results
 
"""

vanilla_gan = load_model('models/generator_v2.h5')
vanilla_hybrid = load_model('models/generator_vanilla3.h5')
wasserstein_gan = load_model('models/generator_v3.h5')
wasserstein_hybrid = load_model('models/generator_vanilla3.h5')

plot_gan_result(vanilla_gan, 2500, 5, 'vanilla_result', True, 'Vanilla GAN Generated Samples', False)
plot_gan_result(vanilla_hybrid, 2500, 48, 'hvanilla_result2', True, 'Hybrid Vanilla  GAN Generated Samples', False)
plot_gan_result(wasserstein_gan, 2500, 48, 'wgan_result', True, 'Wasserstein GAN Generated Samples', True)
plot_gan_result(wasserstein_hybrid, 2500, 48, 'hwgan_result2', True, 'Hybrid Wasserstein GAN Generated Samples', False)


prices1 = pd.DataFrame(generate_samples(vanilla_gan, 2500, 5)[:,1])
prices2 = pd.DataFrame(generate_samples(vanilla_hybrid, 2500, 48)[:,1])[::-1]
prices3 = pd.DataFrame(generate_samples(wasserstein_gan, 2500, 5)[:,1])[::-1]
prices4 = pd.DataFrame(generate_samples(wasserstein_hybrid, 2500, 48)[:,1])[::-1]
real_prices = pd.DataFrame(generate_sp_samples(2500)[0][:,1])

lrets1 = np.log(prices1) - np.log(prices1.shift(1))
lrets2 = np.log(prices2) - np.log(prices2.shift(1))
lrets3 = np.log(prices3) - np.log(prices3.shift(1))
lrets4 = np.log(prices4) - np.ldef group_by_owners(files):
    owners = []
    for n in files.keys:
        
    return None

if __name__ == "__main__":    
    files = {
        'Input.txt': 'Randy',
        'Code.py': 'Stan',
        'Output.txt': 'Randy'
    }   
    print(group_by_owners(files))og(prices4.shift(1))
lrets = np.log(real_prices) - np.log(real_prices.shift(1))


plot_dist(prices1, 'Vanilla GAN Log Return Distribution', True, 'vanilladist')
plot_dist(prices2, 'Hybrid Vanilla GAN Log Return Distribution', True, 'hvdist')
plot_dist(prices4, 'Wasserstein GAN Log Return Distribution', True, 'wdist')
plot_dist(prices3, 'Hybrid Wasserstein GAN Log Return Distribution', True, 'hwdist')

gen_prices= [prices1, prices2, prices3, prices4]
prices = [prices1, prices2, prices3, prices4, real_prices]

"""

Calculating entropies

"""
entropies = []
for price in gen_prices:
    entropies.append(entropy(price, real_prices))

    

""" Plotting Training Histories """

vanilla = pd.read_csv('models/history')
vanilla_h = pd.read_csv('models/vanilla6history')
wasserstein= pd.read_csv('models/wganhistsc')
wasserstein_h = pd.read_csv('models/hwganhist')

plot_gan_history(vanilla.T, title='Vanilla GAN Training History', save=False, fname='vgantraining')
plot_gan_history(vanilla_h.T, title='Hybrid Vanilla GAN Training History', save=False, fname='hvgantraining')
plot_gan_history(wasserstein, title='Wasserstein GAN Training History', save=True, fname='wgantraining')
plot_gan_history(wasserstein_h.T, title='Hybrid Wasserstein GAN Training History', save=True, fname='hwgantraining')



