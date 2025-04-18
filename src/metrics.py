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
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import warnings

def ABPC( y_pred, y_gt, z_values, bw_method = "scott", sample_n = 5000 ):
    """
    Compute the Area Between Probability Curves (ABPC) for two groups.

    Parameters:
    y_pred (np.ndarray): Predicted values.
    y_gt (np.ndarray): Ground truth values.
    z_values (np.ndarray): Binary values indicating the group membership of each sample.
    bw_method (str): Bandwidth method for the kernel density estimation (default is "scott").
    sample_n (int): Number of samples to generate for the integration (default is 5000).

    Returns:
    float: The computed ABPC value.
    """
    # Flatten the input arrays
    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    z_values = z_values.ravel()

    # Extract the predicted values for each group
    y_pre_1 = y_pred[z_values == 1]
    y_pre_0 = y_pred[z_values == 0]

    # Compute the kernel density estimation (KDE) for each group
    kde0 = gaussian_kde(y_pre_0, bw_method = bw_method)
    kde1 = gaussian_kde(y_pre_1, bw_method = bw_method)

    # Generate a set of x values for the integration
    x = np.linspace(0, 1, sample_n)

    # Evaluate the KDEs at the x values
    kde1_x = kde1(x)
    kde0_x = kde0(x)

    # Compute the area between the two KDEs using the trapezoidal rule
    abpc = np.trapz(np.abs(kde0_x - kde1_x), x)

    # Return the computed ABPC value
    return abpc

def ABCC( y_pred, y_gt, z_values, sample_n = 10000 ):
    """
    Compute the Area Between Cumulative Curves (ABCC) for two groups.

    Parameters:
    y_pred (np.ndarray): Predicted values.
    y_gt (np.ndarray): Ground truth values.
    z_values (np.ndarray): Binary values indicating the group membership of each sample.
    sample_n (int): Number of samples to generate for the integration (default is 10000).

    Returns:
    float: The computed ABCC value.
    """
    # Flatten the input arrays
    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    z_values = z_values.ravel()

    # Extract the predicted values for each group
    y_pre_1 = y_pred[z_values == 1]
    y_pre_0 = y_pred[z_values == 0]

    # Compute the empirical cumulative distribution function (ECDF) for each group
    ecdf0 = ECDF(y_pre_0)
    ecdf1 = ECDF(y_pre_1)

    # Generate a set of x values for the integration
    x = np.linspace(0, 1, sample_n)

    # Evaluate the ECDFs at the x values
    ecdf0_x = ecdf0(x)
    ecdf1_x = ecdf1(x)

    # Compute the area between the two ECDFs using the trapezoidal rule
    abcc = np.trapz(np.abs(ecdf0_x - ecdf1_x), x)

    # Return the computed ABCC value
    return abcc


def ABPC_local(y_pred, y_gt, z_values, bw_method = "scott", sample_n = 5000, a=0.0, b=1.0, threshold_based=True ):
    """
    Compute the Area Between Cumulative Curves (ABCC) for two groups.

    Parameters:
    y_pred (np.ndarray): Predicted values.
    y_gt (np.ndarray): Ground truth values.
    z_values (np.ndarray): Binary values indicating the group membership of each sample.
    sample_n (int): Number of samples to generate for the integration (default is 10000).

    Returns:
    float: The computed ABCC value.
    """
    # Flatten the input arrays
    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    z_values = z_values.ravel()

    # Extract the predicted values for each group
    y_pre_1 = y_pred[z_values == 1]
    y_pre_0 = y_pred[z_values == 0]

    # Compute the kernel density estimation (KDE) for each group
    kde0 = gaussian_kde(y_pre_0, bw_method = bw_method)
    kde1 = gaussian_kde(y_pre_1, bw_method = bw_method)

    x = np.linspace(a, b, 2500)

    # Evaluate the KDEs at the x values
    kde1_x = kde1(x)
    kde0_x = kde0(x)

    # Compute the area between the two KDEs using the trapezoidal rule
    abpc_local = np.trapz(np.abs(kde0_x - kde1_x), x)

    # Return the computed ABPC value
    return abpc_local

def ABCC_local(y_pred, y_gt, z_values, sample_n=10000, a=0.0, b=1.0, threshold_based=True):
    """
    Compute the local Area Between Cumulative Curves (ABCC) for two groups.

    Parameters:
    y_pred (np.ndarray): Predicted values.
    y_gt (np.ndarray): Ground truth values.
    z_values (np.ndarray): Binary values indicating the group membership of each sample.
    sample_n (int): Number of samples to generate for the integration (default is 10000).
    a (float): Lower bound of the local area.
    b (float): Upper bound of the local area.
    threshold_based (bool): Whether to use threshold-based evaluation.

    Returns:
    float: The computed local ABCC value.
    """
    # Flatten the input arrays
    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    z_values = z_values.ravel()

    # Extract the predicted values for each group
    y_pre_1 = y_pred[z_values == 1]
    y_pre_0 = y_pred[z_values == 0]

    # Compute the empirical cumulative distribution function (ECDF) for each group
    ecdf0 = ECDF(y_pre_0)
    ecdf1 = ECDF(y_pre_1)

    x = np.linspace(a, b, int(sample_n))

    # Evaluate the ECDFs at the x values
    ecdf0_x = ecdf0(x)
    ecdf1_x = ecdf1(x)

    # Compute the area between the two ECDFs using the trapezoidal rule
    abcc_local = np.trapz(np.abs(ecdf0_x - ecdf1_x), x)

    # Return the computed ABCC value
    return abcc_local

def demographic_parity(y_pred, z_values, threshold=0.5):
    """
    Compute the demographic parity for two groups.

    Parameters:
    y_pred (np.ndarray): Predicted values.
    z_values (np.ndarray): Binary values indicating the group membership of each sample.
    threshold (float): Threshold value for the predicted values (default is 0.5).

    Returns:
    float: The computed demographic parity value.
    """
    # Extract the predicted values for each group and apply the threshold if it is not None
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]

    # Compute the absolute difference between the mean predicted value for each group
    y_z_1_mean = y_z_1.mean()
    y_z_0_mean = y_z_0.mean()
    parity = abs(y_z_1_mean - y_z_0_mean)

    # Return the computed demographic parity value
    return parity

def demographic_parity_c(y_pred, z_values, decision_threshold=0):
    """
    Compute the demographic parity for two groups with a custom decision threshold.

    Parameters:
    y_pred (np.ndarray): Predicted values.
    z_values (np.ndarray): Binary values indicating the group membership of each sample.
    decision_threshold (float): Threshold value for the predicted values (default is 0).

    Returns:
    float: The computed demographic parity value.
    """
    # Extract the predicted values for each group and apply the threshold
    y_z_0 = y_pred[z_values == 0] >= decision_threshold
    y_z_1 = y_pred[z_values == 1] >= decision_threshold

    # Compute the absolute difference between the mean predicted value for each group
    if len(y_z_0) == 0:
        y_z_0_mean = np.nan
    else:
        y_z_0_mean = y_z_0.mean()
    if len(y_z_1) == 0:
        y_z_1_mean = np.nan
    else:
        y_z_1_mean = y_z_1.mean()
    parity = abs(y_z_1_mean - y_z_0_mean)

    # Return the computed demographic parity value
    return parity

def demographic_parity_b(y_pred, z_values, decision_threshold=0.5):
    """
    Compute the demographic parity for two groups with a binary decision threshold.

    Parameters:
    y_pred (np.ndarray): Predicted values.
    z_values (np.ndarray): Binary values indicating the group membership of each sample.
    decision_threshold (float): Threshold value for the predicted values (default is 0.5).

    Returns:
    float: The computed demographic parity value.
    """

    # Extract the predicted values for each group
    y_z_0 = y_pred[z_values == 0]
    y_z_1 = y_pred[z_values == 1]

    # Transform to binary vectors based on decision_threshold by element-wise comparisons
    y_z_0_b = (y_z_0 >= decision_threshold).astype(int)
    y_z_1_b = (y_z_1 >= decision_threshold).astype(int)

    # Compute the absolute difference of the mean binary prediction for each group
    y_z_0_b_mean = y_z_0_b.mean()
    y_z_1_b_mean = y_z_1_b.mean()
    parity = abs(y_z_1_b_mean - y_z_0_b_mean)

    # Return the computed demographic parity value
    return parity

def demographic_parity_local(y_pred, z_values, binary_threshold=0.5, a=0.0, b=1.0, threshold_based=True):
    """
    Compute the local demographic parity for two groups.

    Parameters:
    y_pred (np.ndarray): Predicted values.
    z_values (np.ndarray): Binary values indicating the group membership of each sample.
    binary_threshold (float): Threshold value for the predicted values (default is 0.5).
    a (float): Lower bound of the local area.
    b (float): Upper bound of the local area.
    threshold_based (bool): Whether to use threshold-based evaluation.

    Returns:
    float: The computed local demographic parity value.
    """
    # Extract the predicted values for each group
    y_pre_1 = y_pred[z_values == 1]
    y_pre_0 = y_pred[z_values == 0]

    y_z_0 = y_pre_0[(a < y_pre_0) & (y_pre_0 <= b)]
    y_z_1 = y_pre_1[(a < y_pre_1) & (y_pre_1 <= b)]

    # Compute the absolute difference between the mean predicted value for each group
    if len(y_z_0) == 0:
        # Handle the case when y_z_1 is empty
        y_z_0_mean = np.nan  # or any other appropriate value or action
    else:
        # Calculate the mean if y_z_0 is not empty
        y_z_0_mean = y_z_0.mean()

    if len(y_z_1) == 0:
        # Handle the case when y_z_1 is empty
        y_z_1_mean = np.nan  # or any other appropriate value or action
    else:
        # Calculate the mean if y_z_1 is not empty
        y_z_1_mean = y_z_1.mean()

    parity = abs(y_z_1_mean - y_z_0_mean)

    # Return the computed demographic parity value
    return parity


def metric_evaluation(y_gt, y_pre, s, prefix="", binary_threshold=0.5,pct_a=0.0,pct_b=1.0, threshold_based=True):
    """
    Evaluate various metrics for the given predictions and ground truth values.

    Parameters:
    y_gt (np.ndarray): Ground truth values.
    y_pre (np.ndarray): Predicted values.
    s (np.ndarray): Binary values indicating the group membership of each sample.
    prefix (str): Prefix for the metric names.
    binary_threshold (float): Threshold value for the predicted values (default is 0.5).
    pct_a (float): Lower bound of the local area.
    pct_b (float): Upper bound of the local area.
    threshold_based (bool): Whether to use threshold-based evaluation.

    Returns:
    dict: A dictionary containing the evaluated metrics.
    """
    y_gt = y_gt.ravel()
    y_pre = y_pre.ravel()
    s = s.ravel()

    accuracy = metrics.accuracy_score(y_gt, y_pre > binary_threshold)
    ap = metrics.average_precision_score(y_gt, y_pre)
    dp = demographic_parity(y_pre, s)
    dpe = demographic_parity(y_pre, s, threshold=None)
    abpc = ABPC(y_pre, y_gt, s)
    abcc = ABCC(y_pre, y_gt, s)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Calculate additional metrics
        auc = metrics.roc_auc_score(y_gt, y_pre)  # * 100

        precision = metrics.precision_score(y_gt, y_pre > binary_threshold)  # * 100
        recall = metrics.recall_score(y_gt, y_pre > binary_threshold)  # * 100

    # Select predictions with y_pre in local area
    local_indices = np.where((y_pre > pct_a) & (y_pre <= pct_b))[0]
    local_y_pre = y_pre[local_indices]
    local_y_gt = y_gt[local_indices]

    # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
    # (add small value to avoid div 0)
    TP_local = max(1e-5, np.sum((local_y_pre >= binary_threshold) & (local_y_gt == 1)))
    FP_local = max(1e-5, np.sum((local_y_pre >= binary_threshold) & (local_y_gt == 0)))
    TN_local = max(1e-5, np.sum((local_y_pre < binary_threshold) & (local_y_gt == 0)))
    FN_local = max(1e-5, np.sum((local_y_pre < binary_threshold) & (local_y_gt == 1)))

    np.seterr(divide='ignore')
    accuracy_local = max(1e-5, (TP_local + TN_local) / (TP_local + FP_local + FN_local + TN_local))
    precision_local = max(1e-5, TP_local / (TP_local + FP_local))
    recall_local = max(1e-5, TP_local / (TP_local + FN_local))
    FPR_local = max(1e-5, FP_local / (FP_local + TN_local))
    TPR_local = max(1e-5, TP_local / (TP_local + FN_local))

    FP_tau = np.sum((y_pre >= pct_a) & (y_gt == 0))
    TN_tau = np.sum((y_pre < pct_a) & (y_gt == 0))
    FPR_tau = max(1e-5, FP_tau / (FP_tau + TN_tau))
    partial_auc = roc_auc_score(y_gt, y_pre, max_fpr=FPR_tau)

    abpc_local = ABPC_local(y_pred=y_pre, y_gt=y_gt, z_values=s, bw_method="scott", sample_n=10000, a=pct_a, b=pct_b,
                            threshold_based=threshold_based)
    abcc_local = ABCC_local(y_pred=y_pre, y_gt=y_gt, z_values=s, sample_n=10000, a=pct_a, b=pct_b,
                            threshold_based=threshold_based)
    dp_local = demographic_parity_local(y_pred=y_pre, z_values=s, binary_threshold=binary_threshold, a=pct_a, b=pct_b,
                                        threshold_based=threshold_based)

    pred_in_dec_area_s0 = np.sum((y_pre[s == 0] >= pct_a) & (y_pre[s == 0] <= pct_b))
    pred_in_dec_area_s1 = np.sum((y_pre[s == 1] >= pct_a) & (y_pre[s == 1] <= pct_b))
    pred_in_dec_area_abs = np.sum((y_pre >= pct_a) & (y_pre <= pct_b))
    pred_in_dec_area_rel = pred_in_dec_area_abs / len(y_pre)

    # recall where binary cutoff (for classification 0/1) is set to the fairness cut-off:
    recall_global_tau = metrics.recall_score(y_gt, (y_pre >= pct_a))

    metric_name = ["accuracy", "ap", "dp", "dpe", "abpc", "abcc", "auc", "precision", "recall",
                   'abpc_local', 'abcc_local', 'dp_local', 'accuracy_local', 'precision_local', 'recall_local',
                   'partial_auc', 'pred_in_dec_area_abs', 'pred_in_dec_area_s0', 'pred_in_dec_area_s1',
                   'pred_in_dec_area_rel', 'recall_global_tau']

    metric_name = [prefix + x for x in metric_name]
    metric_val = [accuracy, ap, dp, dpe, abpc, abcc, auc, precision, recall,
                  abpc_local, abcc_local, dp_local, accuracy_local, precision_local, recall_local,
                  partial_auc, pred_in_dec_area_abs, pred_in_dec_area_s0, pred_in_dec_area_s1,
                  pred_in_dec_area_rel, recall_global_tau]

    return dict(zip(metric_name, metric_val))

# Local metrics predictive performance
def local_predictive_performance(y_pred, y_gt, decision_threshold=0.5):
    """
    Compute local predictive performance metrics.

    Parameters:
    y_pred (np.ndarray): Predicted values.
    y_gt (np.ndarray): Ground truth values.
    decision_threshold (float): Threshold value for the predicted values (default is 0.5).

    Returns:
    dict: A dictionary containing the local predictive performance metrics.
    """
    y_gt = y_gt.ravel()
    y_pred = y_pred.ravel()

    # Select observations with a predicted score in the decision range
    local_indices = np.asarray(y_pred >= decision_threshold).nonzero()[0]
    local_y_pred = y_pred[local_indices]
    local_y_gt = y_gt[local_indices]

    # Local performance metrics
    FP_tau = np.sum((y_pred >= decision_threshold) & (y_gt == 0))
    TN_tau = np.sum((y_pred < decision_threshold) & (y_gt == 0))
    FPR_tau = max(1e-5, FP_tau / (FP_tau + TN_tau))

    local_perf = {}
    local_perf['partial_auc'] = metrics.roc_auc_score(y_gt, y_pred, max_fpr=FPR_tau)
    local_perf['precision_local'] = metrics.precision_score(local_y_gt, (local_y_pred >= decision_threshold),
                                                            zero_division=0)
    local_perf['recall_local'] = metrics.recall_score(local_y_gt, (local_y_pred >= decision_threshold),
                                                      zero_division=0)

    return local_perf
