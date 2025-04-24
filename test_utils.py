import numpy as np
from utils import *
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})
from sklearn.svm import SVC
    
# Plot directory
directory = "./results-plot"

fontsize = 30
labelsize = 25

# Parameters
n = 1000
p = 2
gamma = 1e-4
# Mean vector
vmu = np.array([-0.7, 0.7])

fig, ax = plt.subplots(1, 3, figsize = (30, 6))
s = 100
alphas = [0, -1, 1]
t = np.linspace(-4, 4, 50)
for i in range(3):
    vk = alphas[i] * vmu
    X_train, y_train = generate_data(n, vmu, vk, mode = 'features')[0]
    # Train linear SVM
    clf = SVC(kernel='linear', C=10.0)
    clf.fit(X_train.T, y_train)  # Transpose since X_train is shape (2, n)

    # Get coefficients
    w = clf.coef_[0]
    b = clf.intercept_[0]

    # Compute decision boundary: w[0] * x + w[1] * y + b = 0  => y = -(w[0]/w[1]) * x - b/w[1]
    fw = - (w[0] / w[1]) * t - b / w[1]

    ax[i].scatter(X_train[0, y_train < 0], X_train[1, y_train < 0], color = 'tab:orange', label = 'class $\mathcal{C}_1$', s = s)
    ax[i].scatter(X_train[0, y_train > 0], X_train[1, y_train > 0], color = 'tab:green', label = 'class $\mathcal{C}_2$', s= s)
    ax[i].grid(True)
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    ax[i].set_xlim(-5, 5)
    ax[i].set_ylim(-5, 5)
    if i > 0:
        ax[i].plot(t, fw, color = 'tab:red', linestyle = '-.', linewidth = 3, label = '')
        ax[i].set_title(f'Knowledge $k = {round(alphas[i])} \\times \mu $', fontsize = fontsize)

ax[0].set_title(f'Original dataset', fontsize = fontsize)
ax[0].legend(fontsize = labelsize)
path = directory + f'/denoising-property-SVM-n-{n}-p-{p}-gamma-{gamma}.pdf'
fig.savefig(path, bbox_inches='tight')