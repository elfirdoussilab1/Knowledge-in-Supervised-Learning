import numpy as np
import matplotlib.pyplot as plt
from utils import *
from tqdm.auto import tqdm
from rmt_results import *
plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})

# Parameters
n = 2000
p = 200

mu = 0.7
mu_orth = 0.7
vmu = np.zeros(p)
vmu[0] = mu

vmu_orth = np.zeros(p)
vmu_orth[1] =mu_orth 

alphas = np.linspace(-3, 3, 100)
betas = np.linspace(-3, 3, 100)

acc_matrix = np.zeros((len(alphas), len(betas)))
i_max, j_max = 0, 0
acc_max = 0
for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):
        vk = alpha * vmu + beta * vmu_orth
        acc_matrix[i, j] = test_accuracy('w_1', n, p, vmu, vk, gamma = 1e-1)
        if acc_matrix[i, j] > acc_max:
            i_max, j_max = i, j
            acc_max = acc_matrix[i, j]

# Plotting heatmap
plt.figure(figsize=(6, 4))
fontsize = 20
labelsize = 18
img = plt.imshow(acc_matrix.T, origin='lower', extent=[-3, 3, -3, 3], aspect='auto', cmap='viridis')

# Add red dot at max location
#plt.plot(alphas[i_max], betas[j_max], 'ro', markersize=8, label='Max Accuracy')
# Optional: Annotate accuracy value
plt.text(alphas[i_max] + 0.1, betas[j_max], f"{acc_max:.2f}", color='white', fontsize=fontsize)
cbar = plt.colorbar(img)
cbar.set_label('Test Accuracy', fontsize=labelsize)       # Label font size
cbar.ax.tick_params(labelsize=labelsize)  
plt.xlabel(r'$\alpha$', fontsize=fontsize)
plt.ylabel(r'$\beta$', fontsize=fontsize)
plt.tick_params(axis='x', which = 'both', labelsize=labelsize)
plt.tick_params(axis='y', which = 'both', labelsize=labelsize)
plt.title('Test Accuracy for $k = \\alpha \mu + \\beta \mu^\perp$', fontsize=16)
plt.grid(False)
plt.tight_layout()
plt.savefig("./study-plot/accuracy_heatmap.png", dpi=300, bbox_inches="tight")