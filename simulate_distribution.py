# This file is used to generate distribution plots on Synthetic data to validate theoretical calculus
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rmt_results import *

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})
    
# Plot directory
directory = "./study-plot"

fontsize = 40
labelsize = 35

# Parameters
p = 200
n = 5000
pi = 0.4
gamma = 1e-3
classifier = 'w_1'
mode = 'features' if classifier == 'w_1' else 'labels'

# Mean vector
mu = 0.7
vmu = np.zeros(p)
vmu[0] = mu

bayes_acc = round(bayes_accuracy(vmu) * 100, 2)

fig, ax = plt.subplots(1, 3, figsize = (30, 6))
if classifier == 'w_1':
    k_norms = [- mu/2 , 0, 2*mu]

else:
    k_norms = [- 2 , 0, 2]

vks = information_vectors(k_norms, mode, p)

for i in range(len(k_norms)):
    # generate dataset
    (X_train, y_train), (X_test, y_test) = generate_data(n, vmu, vks[i], mode, pi)
    # Classifier
    w = classifier_vector(X_train, y_train, gamma)

    # Expectation of class C_1 and C_2
    mean_c2 = test_expectation(classifier, n, p, vmu, vks[i], gamma)
    mean_c1 = - mean_c2
    expec_2 = test_expectation_2(classifier, n, p, vmu, vks[i], gamma)
    std = np.sqrt(expec_2 - mean_c2**2)

    t1 = np.linspace(mean_c1 - 4*std, mean_c1 + 5*std, 100)
    t2 = np.linspace(mean_c2 - 4*std, mean_c2 + 5*std, 100)
    
    # Plot all
    ax[i].plot(t1, gaussian(t1, mean_c1, std), color = 'tab:red', linewidth= 3)
    ax[i].plot(t2, gaussian(t2, mean_c2, std), color = 'tab:blue', linewidth= 3)
    ax[i].set_xlabel('$\\mathbf{w}^\\top \\mathbf{x}$', fontsize = fontsize)
    ax[i].set_title(rf'{classifier.capitalize()}', fontsize= fontsize)

    # Plotting histogram
    ax[i].hist(X_test[:, :n].T @ w, color = 'tab:red', density = True, bins=25, alpha=.5, edgecolor = 'black')
    ax[i].hist(X_test[:, n:].T @ w, color = 'tab:blue', density = True, bins=25, alpha=.5, edgecolor = 'black')
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    # Label: label = '$\mathcal{C}_2$'
    acc = test_accuracy(classifier, n, p, vmu, vks[i], gamma)
    if classifier == 'w_1':
        ax[i].set_title(f'$k = {round(k_norms[i] / mu, 2)} \\times \mu $, Acc = {round(acc * 100, 2)} \%', fontsize= fontsize)
        ax[i].set_ylim(0, 1.2)
        ax[i].set_xlim(-2, 2)
    else:
        ax[i].set_title(f'$k = {k_norms[i]}$, Acc = {round(acc * 100, 2)} \%', fontsize= fontsize)
        ax[i].set_ylim(0, 0.9)
    
    

ylabel = f'Feature Information' if classifier == 'w_1' else f'Label Information'

ax[0].set_ylabel(ylabel, fontsize = labelsize)
fig.subplots_adjust(top=0.75)
fig.suptitle(f"Bayes Accuracy = {bayes_acc} \%", fontsize = fontsize)
#fig.tight_layout(rect=[0, 0.03, 1, 0.95])
path = directory + f'/distribution-{classifier}-n-{n}-p-{p}-pi-{pi}-mu-{mu}-gamma-{gamma}.pdf'
fig.savefig(path, bbox_inches='tight')
