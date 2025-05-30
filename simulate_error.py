# This file can be used to generate plots of risks that prove that our theory matches empirical quantities
import numpy as np
import matplotlib.pyplot as plt
from rmt_results import *
from utils import *
from tqdm.auto import tqdm

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})

# Parameters
classifier = 'w_1' # classifier = 'w_1' or 'w_2'
mode = 'features' if classifier == 'w_1' else 'labels'
print(mode)
n = 2000
p = 200
pi = 0.4

# Vectors vmu and vk
mu = 1.
vmu = np.zeros(p)
vmu[0] = mu

batch = 20

gammas = np.logspace(-6, 2, 20)

linewidth = 5
fontsize = 40
labelsize = 35
s = 200
alpha = .9

# Expectation
fig, ax = plt.subplots(1, 3, figsize = (30, 6))
k_norms = [0.5, 1.5, 2]
for i, norm_k in enumerate(k_norms):
    vk = np.zeros(p)
    vk[0] = norm_k
    #vk = norm_k
    error_th = []
    error = []
    for gamma in tqdm(gammas):
        # Theory
        error_th.append(test_expectation(classifier, n, p, vmu, vk, gamma))

        # Practice
        error.append(empirical_mean(batch, n, vmu, vk, gamma, pi, mode, data_type = 'synthetic'))
    
    ax[i].semilogx(gammas, error_th, linewidth = linewidth, label = 'Theory', color = 'tab:red')
    ax[i].scatter(gammas, error, s = s , color = 'tab:green', alpha = alpha, marker = 'D', label = 'Simulation')
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    ax[i].set_xlabel('$\gamma$', fontsize = fontsize)
    ax[i].set_title(f'$ \| k \| = {norm_k} $', fontsize = fontsize)
    ax[i].grid(True)

ax[0].set_ylabel('Test Error', fontsize = fontsize)
ax[0].legend(fontsize = labelsize)

path = './results-plot/' + f'simulate-error-{classifier}-n-{n}-p-{p}-pi-{pi}-mu-{mu}.pdf'
fig.savefig(path, bbox_inches='tight')
