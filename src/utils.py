from datetime import datetime

import numpy as np
import pandas as pd
import torch
import wandb
from matplotlib import pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
from numpy import random

import os

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from aif360.datasets import BinaryLabelDataset

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy.stats import gaussian_kde

from statsmodels.distributions.empirical_distribution import ECDF

import logging

import seaborn as sns

from src.data import preprocess_adult_data, preprocess_churn_data, preprocess_telecomkaggle_data
from src.loss import Wasserstein_reg, dp_reg
from src.metrics import metric_evaluation
from src.model import MLP, Logit



def plot_intermediate_steps(pre_clf_test, y_test, s_test_sex, test_metric, fair_loss, pct_a, pct_b, args, epoch):

    # Set the style of the plot
    plt.style.use('science')

    y_pre_1 = pre_clf_test[s_test_sex.flatten() == 1]
    y_pre_0 = pre_clf_test[s_test_sex.flatten() == 0]

    fig, ax1 = plt.subplots()

    sns.kdeplot(y_pre_0, color='blue', label='s=0', ax=ax1)
    sns.kdeplot(y_pre_1, color='red', label='s=1', ax=ax1)

    ax1.set_ylabel('Probability density', color='black')

    # Add labels and title
    plt.xlabel('Predicted score')
    plt.xlim(0, 1)

    # Add a legend
    ax1.legend(frameon=True, framealpha=0.5)

    if args['threshold_based']:

        # Plot decision area
        plt.axvline(pct_a, color='grey', linestyle=':')
        # on x-axis, put $\tau$ at (pct_a, 0):
        plt.text(pct_a, -0.0, r'$\tau$', verticalalignment='top', horizontalalignment='center', fontsize=14)

    plt.show()


    """ Code for plotting the PR curve (full and local)"""
    # Plot PR curve
    ##### local AUC-PR for different precision thresholds

    # define precision_at_tau:
    precision_at_tau = precision_score(y_test, (pre_clf_test >= pct_a).astype(int))

    # Compute precision-recall curve
    precision_auc_pr, recall_auc_pr, _ = precision_recall_curve(y_test, pre_clf_test)

    # plot the PR curve
    plt.plot(recall_auc_pr, precision_auc_pr, marker='.', label='PR curve')

    # plot the local PR curve:
    # Select indices where precision is at least precision_min
    valid_indices = np.where(precision_auc_pr >= precision_at_tau)[0]
    precision_partial = precision_auc_pr[valid_indices]
    recall_partial = recall_auc_pr[valid_indices]

    # Compute partial AUC-PR using the trapezoidal rule
    partial_auc_pr = auc(recall_partial, precision_partial)

    # Compute full AUC-PR
    full_auc_pr = auc(recall_auc_pr, precision_auc_pr)

    print("Partial AUC-PR (Precision above", precision_at_tau, "):", partial_auc_pr)
    print("Full AUC-PR:", full_auc_pr)

    # Plot the precision-recall curve
    plt.figure()
    plt.plot(recall_auc_pr, precision_auc_pr, color='black', lw=2, label='PR curve (AUC-PR = %0.2f)' % full_auc_pr)
    plt.plot(recall_partial, precision_partial, color='red', lw=2, linestyle='--',
             label=r'Partial PR curve ($\text{AUC-PR}_\tau$ = %0.2f)' % partial_auc_pr)
    plt.fill_between(recall_partial, precision_partial, step='post', alpha=0.1, color='red')

    recall_at_precision = recall_auc_pr[np.argmax(precision_auc_pr >= precision_at_tau)]
    plt.plot(recall_at_precision, precision_at_tau, marker='o', markersize=8, color='red', label='Decision area cutoff')
    plt.plot([recall_at_precision, recall_at_precision], [0, precision_at_tau], color='grey', linestyle='--')
    plt.plot([0, recall_at_precision], [precision_at_tau, precision_at_tau], color='grey', linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(fr'Partial PR curve for $\tau={pct_a}$')
    plt.legend(loc="lower left", fontsize=8)

    plt.show()


class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_result_dict(epoch_dict):
    # List of keys to keep in the subset
    keys_to_keep = ['data_path','sensitive_attr','method','MLP_n_hidden_layers','MLP_hidden_size','MLP_p_dropout',
                    'num_epochs','batch_size','lr','seed','lam','fair_reg','local_reg','pct_a','pct_b','epoch','ce_loss',
                    'f_loss','test_accuracy','test_ap','test_dp','test_dpe','test_abpc','test_abcc','test_auc',
                    'test_precision','test_recall','test_abpc_local','test_abcc_local','test_dp_local']

    # Create a new dictionary containing only the specified keys
    subset_dict = {key: epoch_dict[key] for key in keys_to_keep}

    # Check if the model is 'Logit' and update MLP architecture information
    if subset_dict.get('method') == 'Logit':
        subset_dict['MLP_hidden_size'] = 1
        subset_dict['MLP_p_dropout'] = 0
        subset_dict['MLP_n_hidden_layers'] = 1


    return subset_dict





