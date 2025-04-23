# This file will be used to generate the values of accuracies in the table 1 in the paper.
from utils import *
from rmt_results import *
import pandas as pd
from tqdm.auto import tqdm
from dataset import *

# Parameters
n = 1600
p = 400
gamma = 100
batch = 50

# Create results in a dataframe: rows = datasets (4), columns = algorithms (3)
data_types = ['book', 'dvd', 'elec', 'kitchen']
results = pd.DataFrame(columns=['k_1', 'std_1', 'k_2', 'std_2', 'k_3', 'std_3'])
seeds = [1, 123, 404]

ks = [-1/2, 0, 3]

for typ in tqdm(data_types):
    data_name = 'amazon_' + typ
    accs_1 = []
    accs_2 = []
    accs_3 = []
    for seed in seeds:
        fix_seed(seed)
        # k[0]
        accs_1.append(empirical_accuracy(batch, n, None, ks[0], gamma, 0.5, 'features', data_name))

        # k[1]
        accs_2.append(empirical_accuracy(batch, n, None, ks[1], gamma, 0.5, 'features', data_name))

        # k[2]
        accs_3.append(empirical_accuracy(batch, n, None, ks[2], gamma, 0.5, 'features', data_name))

    row = {'k_1': round(np.mean(accs_1) * 100, 2),
        'std_1' : round(np.std(accs_1) * 100, 2),
        'k_2' : round(np.mean(accs_2) * 100, 2),
        'std_2' : round(np.std(accs_2) * 100, 2),
        'k_3': round(np.mean(accs_3) * 100, 2), 
        'std_3' : round(np.std(accs_3) * 100, 2),
        }
    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

results.index = data_types
path = f'accuracy_amazon-n-{n}-p-{p}-gamma-{gamma}.csv'
results.to_csv(path)
