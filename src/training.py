# Standard library imports
import datetime
import logging

# Third-party library imports
import wandb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Project-specific imports
from src.data import (
    preprocess_adult_data,
    preprocess_churn_data,
    preprocess_telecomkaggle_data,
)
from src.loss import (
    dp_reg,
    Wasserstein_reg
)
from src.metrics import metric_evaluation
from src.model import MLP, Logit
from src.utils import PandasDataSet, plot_intermediate_steps

def train_experiment(experiment, counter, total_combinations):
    """
    Train a model based on the given experiment configuration.

    Parameters:
    experiment (dict): Configuration dictionary for the experiment.
    counter (int): Current experiment number.
    total_combinations (int): Total number of experiment combinations.

    Returns:
    None
    """
    start_time = datetime.datetime.now()
    print('\n' + datetime.datetime.now().strftime('Experiment started at:  %d-%m-%y  |  %H:%M'))
    print(f"*Experiment {counter}/{total_combinations}* Data: {experiment['data_path']}({experiment['semi_synth_bias']}_{experiment['bias_ratio']}), local_reg: {experiment['local_reg']}, fair_reg: {experiment['fair_reg']}, lam: {experiment['lam']}")

    # Initialize Wandb
    if experiment['enableWandb']:
        wandb.init(project=experiment['wandb_project'], entity='simondevos', reinit=False, save_code=True)

    epoch_dict = experiment.copy()

    # Delete previously trained classifier
    try:
        del clf
        print("previously initialized 'clf' deleted.")
    except NameError:
        pass

    # Set random Torch seed for reproducibility
    torch.manual_seed(experiment['seed'])

    # Define the training loop function
    def train(model, train_loader, optimizer, criterion, device):
        """
        Train the model for one epoch.

        Parameters:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the training on (CPU or GPU).

        Returns:
        float: The average training loss for the epoch.
        """
        model.train()
        running_loss = 0.0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    # Set logging level to WARNING
    logging.basicConfig(level=logging.WARNING)
    # Initialize logger object
    logger = logging.getLogger()

    # Clear existing handlers (if any)
    logger.handlers = []  # otherwise, the logger keeps getting longer per run

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Set device (CPU or GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if experiment['log_screen'] is True:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Prepare the data
    if experiment['data_path'] == "data/adult":
        logger.info(f'Dataset: adult')
        X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test = preprocess_adult_data(
            seed=experiment['seed'], path=experiment['data_path'], sensitive_attributes=experiment['sensitive_attr'],
            train_size=experiment['train_size'])

    elif experiment['data_path'] == 'data/churn':
        logger.info(f'Dataset: churn')
        X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test = preprocess_churn_data(
            seed=experiment['seed'], path=experiment['data_path'], sensitive_attributes=experiment['sensitive_attr'])

    elif experiment['data_path'] == 'data/TelecomKaggle':
        logger.info(f'Dataset: TelecomKaggle')
        X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test = preprocess_telecomkaggle_data(
            seed=experiment['seed'], path=experiment['data_path'], sensitive_attributes=experiment['sensitive_attr'],
            semi_synth_bias=experiment['semi_synth_bias'], bias_ratio=experiment['bias_ratio'],
            train_size=experiment['train_size'])



    else:
        logger.info(f'Wrong dataset_name')
        raise ValueError(
            "Warning: only datasets 'adult', 'churn', 'TelecomKaggle' are implemented")

    X = pd.DataFrame(np.concatenate([X_train, X_val, X_test]))
    y = pd.DataFrame(np.concatenate([y_train, y_val, y_test]))[0]
    s = pd.DataFrame(np.concatenate([A_train, A_val, A_test]))

    X_train = pd.DataFrame(X_train)
    X_val = pd.DataFrame(X_val)
    X_test = pd.DataFrame(X_test)

    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    y_test = pd.DataFrame(y_test)

    s_train = pd.DataFrame(A_train)
    s_val = pd.DataFrame(A_val)
    s_test = pd.DataFrame(A_test)

    logger.info(f'X.shape: {X.shape}')
    logger.info(f'y.shape: {y.shape}')
    logger.info(f's.shape: {s.shape}')
    logger.info(f's.shape: {s.value_counts().to_dict()}')

    n_features = X.shape[1]

    logger.info(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}, s_train.shape: {s_train.shape}')
    logger.info(f'X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}, s_val.shape: {s_val.shape}')
    logger.info(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}, s_test.shape: {s_test.shape}')

    scaler = StandardScaler().fit(X_train)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    X_train = X_train.pipe(scale_df, scaler)
    X_val = X_val.pipe(scale_df, scaler)
    X_test = X_test.pipe(scale_df, scaler)

    train_data = PandasDataSet(X_train, y_train, s_train)
    val_data = PandasDataSet(X_val, y_val, s_val)
    test_data = PandasDataSet(X_test, y_test, s_test)

    train_loader = DataLoader(train_data, batch_size=experiment['batch_size'], shuffle=True)
    train_loader_no_shuffle = DataLoader(train_data, batch_size=experiment['batch_size'], shuffle=False)
    val_loader = DataLoader(val_data, batch_size=experiment['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=experiment['batch_size'], shuffle=False)

    s_val = s_val.values
    y_val = y_val.values

    s_test = s_test.values
    y_test = y_test.values

    s_train = s_train.values
    y_train = y_train.values

    epoch_dict = experiment.copy()

    # Assign whether MLP or logistic regression is used
    if experiment['method'] == 'MLP':
        clf = MLP(n_features=n_features,
                  n_hidden_layers=experiment['MLP_n_hidden_layers'],
                  hidden_size=experiment['MLP_hidden_size'],
                  p_dropout=experiment['MLP_p_dropout']).to(device)
    elif experiment['method'] == 'Logit':
        clf = Logit(n_features=n_features)
    else:
        print("no valid method specified")

    # Assign the accuracy-based loss component
    clf_criterion = nn.BCELoss()

    # Assign the fairness-based loss component
    local_reg = experiment['local_reg']

    if experiment['fair_reg'] == 1:
        fair_loss = dp_reg(local_reg=local_reg, threshold_based=experiment['threshold_based'])

    elif experiment['fair_reg'] == 2:
        fair_loss = Wasserstein_reg(local_reg=local_reg, threshold_based=experiment['threshold_based'],
                                    top_percentile=experiment['top_percentile'])

    # Assign optimizer
    clf_optimizer = optim.Adam(clf.parameters(), lr=experiment['lr'], weight_decay=experiment['l2_reg'])

    # Training loop
    # Initialize variables for early stopping
    best_composite_loss = float('inf')
    epochs_without_improvement = 0
    early_stopping_patience = experiment['early_stopping_patience']
    early_stopped = 0

    for epoch in range(experiment['num_epochs']):
        lam = experiment['lam']
        ce_loss_train_list = []
        f_loss_train_list = []

        for x, y, s in train_loader:
            x = x.to(device)
            y = y.to(device)
            clf.zero_grad()
            p_y = clf(x)
            ce_loss = clf_criterion(p_y, y)

            f_loss = torch.tensor(0)

            # check for reg delay (first n epochs without regularization)
            if epoch > experiment['reg_delay'] - 1:
                f_loss, _, _, _ = fair_loss(p_y.reshape(-1), s.reshape(-1), y.reshape(-1),
                                            pct_a=experiment['pct_a'], pct_b=experiment['pct_b'])
                loss = (1 - lam) * ce_loss + lam * f_loss

            else:
                loss = ce_loss

            ce_loss_train_list.append(ce_loss.item())
            f_loss_train_list.append(f_loss.item())

            loss.backward()
            clf_optimizer.step()

        epoch_dict["epoch"] = epoch
        epoch_dict["ce_loss_train"] = np.mean(ce_loss_train_list)
        epoch_dict["f_loss_train"] = np.mean(f_loss_train_list)
        epoch_dict["composite_loss_train"] = (1 - lam) * epoch_dict["ce_loss_train"] + lam * epoch_dict["f_loss_train"]

        # validate train_data
        with torch.no_grad():
            y_pre_list = []
            for x, y, s in train_loader_no_shuffle:
                x = x.to(device)
                y = y.to(device)
                p_y = clf(x)
                y_pre_list.append(p_y[:, 0].data.cpu().numpy())

        pre_clf_train = np.concatenate(y_pre_list)
        s_train_sex = s_train

        train_metric = metric_evaluation(y_gt=y_train, y_pre=pre_clf_train, s=s_train_sex,
                                         prefix="train_", binary_threshold=experiment['pct_a'], # set binary cut-off equal to decision area \tau
                                         pct_a=experiment['pct_a'], pct_b=experiment['pct_b'],
                                         threshold_based=experiment['threshold_based'])
        epoch_dict.update(train_metric)

        ce_loss_val_list = []
        f_loss_val_list = []

        # Validate val_data
        with torch.no_grad():
            y_pre_list = []
            for x, y, s in val_loader:
                x = x.to(device)
                y = y.to(device)
                p_y = clf(x)
                y_pre_list.append(p_y[:, 0].data.cpu().numpy())
                loss_val = clf_criterion(p_y, y)

                ce_loss_val_list.append(loss_val.item())

                if epoch > experiment['reg_delay'] - 1:
                    f_loss, _, _, _ = fair_loss(p_y.reshape(-1), s.reshape(-1), y.reshape(-1),
                                                pct_a=experiment['pct_a'], pct_b=experiment['pct_b'])
                    f_loss_val_list.append(f_loss.item())

        epoch_dict['ce_loss_val'] = np.mean(ce_loss_val_list)
        epoch_dict['f_loss_val'] = np.mean(f_loss_val_list) if f_loss_val_list else 0
        epoch_dict['composite_loss_val'] = (1 - lam) * epoch_dict['ce_loss_val'] + lam * epoch_dict['f_loss_val']

        pre_clf_val = np.concatenate(y_pre_list)
        s_val_sex = s_val

        val_metric = metric_evaluation(y_gt=y_val, y_pre=pre_clf_val, s=s_val_sex,
                                       prefix="val_", binary_threshold=experiment['pct_a'],
                                       pct_a=experiment['pct_a'], pct_b=experiment['pct_b'],
                                       threshold_based=experiment['threshold_based'])
        epoch_dict.update(val_metric)

        ce_loss_test_list = []
        f_loss_test_list = []

        # Validate test_data
        with torch.no_grad():
            y_pre_list = []
            for x, y, s in test_loader:
                x = x.to(device)
                y = y.to(device)
                p_y = clf(x)
                y_pre_list.append(p_y[:, 0].data.cpu().numpy())

                loss_test = clf_criterion(p_y, y)
                ce_loss_test_list.append(loss_test.item())

                if epoch > experiment['reg_delay'] - 1:
                    f_loss, _, _, _ = fair_loss(p_y.reshape(-1), s.reshape(-1), y.reshape(-1),
                                                pct_a=experiment['pct_a'], pct_b=experiment['pct_b'])
                    f_loss_test_list.append(f_loss.item())

        epoch_dict["ce_loss_test"] = np.mean(ce_loss_test_list)
        epoch_dict['f_loss_test'] = np.mean(f_loss_test_list) if f_loss_test_list else 0
        epoch_dict['composite_loss_test'] = (1 - lam) * epoch_dict['ce_loss_test'] + lam * epoch_dict['f_loss_test']

        pre_clf_test = np.concatenate(y_pre_list)
        s_test_sex = s_test

        test_metric = metric_evaluation(y_gt=y_test, y_pre=pre_clf_test, s=s_test_sex,
                                        prefix="test_", binary_threshold=experiment['pct_a'],
                                        pct_a=experiment['pct_a'], pct_b=experiment['pct_b'],
                                        threshold_based=experiment['threshold_based'])
        epoch_dict.update(test_metric)

        # one epoch
        logger.info(f"epoch_dict: {epoch_dict}")

        result_dict = epoch_dict

        if experiment['plot_result_per_epoch']:
            plot_intermediate_steps(pre_clf_test, s_test_sex, test_metric, fair_loss,
                                    experiment['pct_a'], experiment['pct_b'], args=experiment, epoch=epoch)

        if experiment['plot_result_last_epoch']:
            if epoch == experiment['num_epochs'] - 1:
                plot_intermediate_steps(pre_clf_test, s_test_sex, test_metric, fair_loss,
                                        experiment['pct_a'], experiment['pct_b'], args=experiment, epoch=epoch)

        # Log experiment metrics to Wandb
        if experiment['enableWandb']:
            wandb.log(result_dict)

        if epoch == experiment['reg_delay']+1:
            logger.info(f"Start regularization after {experiment['reg_delay']} epochs")
            best_composite_loss = float('inf')

        # Early stopping
        if result_dict['composite_loss_val'] < best_composite_loss:
            best_composite_loss = result_dict['composite_loss_val']
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch} epochs")
                early_stopped = 1


                if experiment['plot_result_last_epoch']:
                    #plot distributions also when early stopping, if desired
                    plot_intermediate_steps(pre_clf_test, s_test_sex, test_metric, fair_loss,
                                            experiment['pct_a'], experiment['pct_b'], args=experiment, epoch=epoch)

                break

    result_dict['early_stopped'] = early_stopped

    if experiment['enableWandb']:
        # Finish logging the experiment
        wandb.finish()
    end_time = datetime.datetime.now()
    print('Training time: ' + str(end_time - start_time))
