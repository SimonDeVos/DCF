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
from sklearn.metrics import accuracy_score
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


def plot_intermediate_steps(pre_clf_test, s_test_sex, test_metric, fair_loss, pct_a, pct_b, args, epoch):

    y_pre_1 = pre_clf_test[s_test_sex.flatten() == 1]
    y_pre_0 = pre_clf_test[s_test_sex.flatten() == 0]

    # Plot the distributions
    sns.kdeplot(y_pre_1, label='s==1')
    sns.kdeplot(y_pre_0, label='s==0')

    # Fill under the distributions

    # Add labels and title
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.title('Epoch '+str(epoch+1)+': Probability Distribution of y_pre_1 and y_pre_0')

    # Set x-axis limits
    plt.xlim(0, 1)

    # Add a legend
    plt.legend()

    # Display test accuracy value
    test_accuracy_value = test_metric['test_accuracy']
    test_abpc_value = test_metric['test_abpc']
    test_abcc_value = test_metric['test_abcc']
    test_auc_value = test_metric['test_auc']
    test_dp_value = test_metric['test_dp']
    test_precision_value = test_metric['test_precision']
    test_recall_value = test_metric['test_recall']

    plt.text(0.7, 0.85, f'Sensitive attribute: {str(args["sensitive_attr"])}', transform=plt.gca().transAxes)
    plt.text(0.7, 0.80, f'Test ABPC: {test_abpc_value:.4f}', transform=plt.gca().transAxes)
    plt.text(0.7, 0.75, f'Test ABCC: {test_abcc_value:.4f}', transform=plt.gca().transAxes)
    plt.text(0.7, 0.70, f'Test dp: {test_dp_value:.4f}', transform=plt.gca().transAxes)

    plt.text(0.7, 0.60, f'Test Accuracy: {test_accuracy_value:.4f}', transform=plt.gca().transAxes)
    plt.text(0.7, 0.55, f'Test AUC: {test_auc_value:.4f}', transform=plt.gca().transAxes)
    plt.text(0.7, 0.50, f'Test prec: {test_precision_value:.4f}', transform=plt.gca().transAxes)
    plt.text(0.7, 0.45, f'Test recall: {test_recall_value:.4f}', transform=plt.gca().transAxes)

    if True:

        plt.text(0.1, 0.80, f'Prct area.: {pct_a:.2f} - {pct_b:.2f}', transform=plt.gca().transAxes)

        test_abpc_local_value = test_metric['test_abpc_local']
        plt.text(0.1, 0.70, f'Test ABPC_local.: {test_abpc_local_value:.4f}', transform=plt.gca().transAxes)
        test_abcc_local_value = test_metric['test_abcc_local']
        plt.text(0.1, 0.65, f'Test ABCC_local.: {test_abcc_local_value:.4f}', transform=plt.gca().transAxes)
        test_abcc_local_value = test_metric['test_dp_local']
        plt.text(0.1, 0.60, f'Test DP_c_local.: {test_abcc_local_value:.4f}', transform=plt.gca().transAxes)

        test_acc_local_value = test_metric['test_accuracy_local']
        plt.text(0.1, 0.5, f'Test accuracy_local.: {test_acc_local_value:.4f}', transform=plt.gca().transAxes)
        test_prec_local_value = test_metric['test_precision_local']
        plt.text(0.1, 0.45, f'Test precision_local.: {test_prec_local_value:.4f}', transform=plt.gca().transAxes)
        test_rec_local_value = test_metric['test_recall_local']
        plt.text(0.1, 0.4, f'Test rec_local.: {test_rec_local_value:.4f}', transform=plt.gca().transAxes)
        test_partial_auc_local_value = test_metric['test_partial_auc']
        plt.text(0.1, 0.35, f'Test partial_auc.: {test_partial_auc_local_value:.4f}', transform=plt.gca().transAxes)

        if args['threshold_based']:
            plt.axvline(pct_a, color='grey', linestyle=':')
            plt.axvline(pct_b, color='grey', linestyle=':')

        else:
            # this draws percentile based on whole pop:
            percentile_a = np.percentile(pre_clf_test, pct_a * 100)
            plt.axvline(percentile_a, linestyle=':', label='pct_a', color='fuchsia')

            # Calculate percentiles
            pct_low_0 = np.percentile(y_pre_0, pct_a*100)
            pct_low_1 = np.percentile(y_pre_1, pct_a*100)
            # Mark the percentiles on the plot
            plt.axvline(pct_low_0, color='orange', linestyle=':')#, label='s==0')
            plt.axvline(pct_low_1, linestyle=':')#, label='s==1')

            # Plot decision area
            plt.axvline(pct_a, color='grey', linestyle=':')
            plt.axvline(pct_b, color='grey', linestyle=':')

    # Show the plot
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





