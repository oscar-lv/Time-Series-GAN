#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:53:04 2020

@author: oscar
"""

from reporting import plot_gan_history, plot_dist, generate_samples
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import numpy as np


vanilla_gan = load_model('models/generator_v2.h5')
vanilla_hybrid = load_model('models/generator_v3.h5')
wasserstein_gan = load_model('models/generator_vanilla2.h5')
wasserstein_hybrid = load_model('models/generator_vanilla3.h5')

prices1 = pd.DataFrame(generate_samples(vanilla_gan, 2500, 5))
prices2 = pd.DataFrame(generate_samples(vanilla_hybrid, 2500, 5))
prices3 = pd.DataFrame(generate_samples(wasserstein_gan, 2500, 5))
prices4 = pd.DataFrame(generate_samples(wasserstein_hybrid, 2500, 48))


vanilla = pd.read_csv('models/history')
vanilla_h = pd.read_csv('models/hganhistory')
wasserstein= pd.read_csv('models/wganhist')
wasserstein_h = pd.read_csv('models/hwganhist')

generator = wasserstein_hybrid
n, latent_dim = 2500, 48
np.random.seed(2333)
noise = np.random.normal(0, 1, (n,latent_dim))
gen_data = generator.predict(noise)
b = gen_data.reshape(n, 2)
fig, axs = plt.subplots()
print("noise shape")
print(noise.shape)
print(noise[0])
axs.scatter(b[:, 0], b[:, 1], color='red', label='generated')
 
plot_gan_history(vanilla.T)
plot_dist(prices2)
