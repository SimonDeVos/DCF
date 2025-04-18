import os
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from aif360.datasets import BinaryLabelDataset

def _quantization_binning(data, num_bins=10):
    """
    Quantize the data into bins.

    Parameters:
    data (np.ndarray): The data to be quantized.
    num_bins (int): The number of bins to use for quantization.

    Returns:
    tuple: A tuple containing bin edges, bin centers, and bin widths.
    """
    qtls = np.arange(0.0, 1.0 + 1 / num_bins, 1 / num_bins)
    bin_edges = np.quantile(data, qtls, axis=0)  # (num_bins + 1, num_features)
    bin_widths = np.diff(bin_edges, axis=0)
    bin_centers = bin_edges[:-1] + bin_widths / 2  # ()
    return bin_edges, bin_centers, bin_widths

def _quantize(inputs, bin_edges, num_bins=10):
    """
    Quantize the inputs based on bin edges.

    Parameters:
    inputs (np.ndarray): The data to be quantized.
    bin_edges (np.ndarray): The edges of the bins.
    num_bins (int): The number of bins.

    Returns:
    np.ndarray: The quantized data.
    """
    quant_inputs = np.zeros(inputs.shape[0])
    for i, x in enumerate(inputs):
        quant_inputs[i] = np.digitize(x, bin_edges)
    quant_inputs = quant_inputs.clip(1, num_bins) - 1  # Clip edges
    return quant_inputs

def _one_hot(a, num_bins=10):
    """
    Convert an array to one-hot encoding.

    Parameters:
    a (np.ndarray): The array to be converted.
    num_bins (int): The number of bins for one-hot encoding.

    Returns:
    np.ndarray: The one-hot encoded array.
    """
    return np.squeeze(np.eye(num_bins)[a.reshape(-1).astype(np.int32)])

def DataQuantize(X, bin_edges=None, num_bins=10):
    """
    Quantize the data. First 4 entries are continuous, and the rest are binary.

    Parameters:
    X (np.ndarray): The data to be quantized.
    bin_edges (np.ndarray, optional): The edges of the bins.
    num_bins (int): The number of bins.

    Returns:
    tuple: Quantized data and bin edges.
    """
    X_ = []
    for i in range(5):
        if bin_edges is not None:
            Xi_q = _quantize(X[:, i], bin_edges, num_bins)
        else:
            bin_edges, bin_centers, bin_widths = _quantization_binning(X[:, i], num_bins)
            Xi_q = _quantize(X[:, i], bin_edges, num_bins)
        Xi_q = _one_hot(Xi_q, num_bins)
        X_.append(Xi_q)

    for i in range(5, len(X[0])):
        if i == 39:     # gender attribute
            continue
        Xi_q = _one_hot(X[:, i], num_bins=2)
        X_.append(Xi_q)

    return np.concatenate(X_ ,1), bin_edges

def get_adult_data(path):
    '''
    We borrow the code from https://github.com/IBM/sensitive-subspace-robustness
    Preprocess the adult data set by removing some features and put adult data into a BinaryLabelDataset
    You need to download the adult dataset (both the adult.data and adult.test files) from https://archive.ics.uci.edu/ml/datasets/Adult

    Load and preprocess the Adult dataset.

    Parameters:
    path (str): The path to the dataset.

    Returns:
    BinaryLabelDataset: The preprocessed dataset.

    '''

    headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-stataus', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'y']

    train = pd.read_csv(os.path.join(path, "adult.data"), header = None, sep=r'\s*,\s*', engine='python')
    test = pd.read_csv(os.path.join(path, "adult.test"), header = None,  sep=r'\s*,\s*', engine='python', skiprows=1)

    df = pd.concat([train, test], ignore_index=True)
    df.columns = headers

    df['y'] = df['y'].replace({'<=50K.': 0, '>50K.': 1, '>50K': 1, '<=50K': 0 })

    df = df.drop(df[(df[headers[-2]] == ' ?') | (df[headers[6]] == ' ?')].index)
    df = pd.get_dummies(df, columns=[headers[1], headers[5], headers[6], headers[7], headers[9], headers[8], 'native-country'])

    delete_these = ['race_Amer-Indian-Eskimo' ,'race_Asian-Pac-Islander' ,'race_Black' ,'race_Other', 'sex_Female']
    delete_these += ['native-country_Cambodia', 'native-country_Canada', 'native-country_China', 'native-country_Columbia', 'native-country_Cuba', 'native-country_Dominican-Republic', 'native-country_Ecuador', 'native-country_El-Salvador', 'native-country_England', 'native-country_France', 'native-country_Germany', 'native-country_Greece', 'native-country_Guatemala', 'native-country_Haiti', 'native-country_Holand-Netherlands', 'native-country_Honduras', 'native-country_Hong', 'native-country_Hungary', 'native-country_India', 'native-country_Iran', 'native-country_Ireland', 'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan', 'native-country_Laos', 'native-country_Mexico', 'native-country_Nicaragua', 'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru', 'native-country_Philippines', 'native-country_Poland', 'native-country_Portugal', 'native-country_Puerto-Rico', 'native-country_Scotland', 'native-country_South', 'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago', 'native-country_United-States', 'native-country_Vietnam', 'native-country_Yugoslavia']
    delete_these += ['fnlwgt', 'education']

    df.drop(delete_these, axis=1, inplace=True)

    return BinaryLabelDataset(df = df, label_names = ['y'], protected_attribute_names = ['sex_Male', 'race_White'])

def preprocess_adult_data(seed = 0, path = "", sensitive_attributes="sex", train_size=0.66):
    """
    Preprocess the Adult dataset.

    Parameters:
    seed (int): Random seed for reproducibility.
    path (str): The path to the dataset.
    sensitive_attributes (str): The sensitive attribute to be used.
    train_size (float): The proportion of the dataset to include in the train split.

    Returns:
    tuple: Training, validation, and test sets along with their labels and sensitive attributes.
    """

    ADULT_GENDER_ATTRIBUTE_INDEX = 39
    ADULT_RACE_ATTRIBUTE_INDEX = 40

    # Get the dataset and split into train and test
    dataset_orig = get_adult_data(path)

    continous_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    continous_features_indices = [dataset_orig.feature_names.index(feat) for feat in continous_features]

    dataset_orig_train, dataset_orig_test = dataset_orig.split([train_size], shuffle=True, seed=seed)
    SS = StandardScaler().fit(dataset_orig_train.features[:, continous_features_indices])
    dataset_orig_train.features[:, continous_features_indices] = SS.transform \
        (dataset_orig_train.features[:, continous_features_indices])
    dataset_orig_test.features[:, continous_features_indices] = SS.transform \
        (dataset_orig_test.features[:, continous_features_indices])

    X_train = dataset_orig_train.features
    X_test = dataset_orig_test.features

    y_train = dataset_orig_train.labels
    y_test = dataset_orig_test.labels

    X_val = X_train[:len(X_test)]
    y_val = y_train[:len(X_test)]
    X_train = X_train[len(X_test):]
    y_train = y_train[len(X_test):]

    if sensitive_attributes == "sex":
        A_train = X_train[: ,ADULT_GENDER_ATTRIBUTE_INDEX]
        A_val = X_val[: ,ADULT_GENDER_ATTRIBUTE_INDEX]
        A_test = X_test[: ,ADULT_GENDER_ATTRIBUTE_INDEX]
    elif sensitive_attributes == "race":
        A_train = X_train[: ,40]
        A_val = X_val[: ,ADULT_RACE_ATTRIBUTE_INDEX]
        A_test = X_test[: ,ADULT_RACE_ATTRIBUTE_INDEX]

    X_train, bin_edges = DataQuantize(X_train)
    X_val, _ = DataQuantize(X_val, bin_edges)
    X_test, _ = DataQuantize(X_test, bin_edges)

    return X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test

def get_churn_data(path):
    """
    Load and preprocess the Churn dataset.

    Parameters:
    path (str): The path to the dataset.

    Returns:
    BinaryLabelDataset: The preprocessed dataset.
    """
    headers = ['SEXC', 'DRIVE', 'JOB', 'AGEC', 'POP', 'POI', 'PREMIUM', 'PAYCNT', 'RECNT', 'DELCNT', 'TARGET']
    df = pd.read_csv(os.path.join(path, "churn.csv"), header=0, sep=',', engine='python')
    df.columns = headers

    df['TARGET'] = df['TARGET'].replace({'N': 0, 'Y': 1})
    df['SEXC'] = df['SEXC'].replace({'Female': 0, 'Male': 1})
    df['DRIVE'] = df['DRIVE'].replace({'No': 0, 'Yes': 1})

    # Encode categorical features
    df = pd.get_dummies(df, columns=['JOB'])

    return BinaryLabelDataset(df=df, label_names=['TARGET'], protected_attribute_names=['SEXC'])

def preprocess_churn_data(seed=0, path="", sensitive_attributes="sex"):
    """
    Preprocess the churn dataset.

    Parameters:
    seed (int): Random seed for reproducibility.
    path (str): The path to the dataset.
    sensitive_attributes (str): The sensitive attribute to be used.

    Returns:
    tuple: Training, validation, and test sets along with their labels and sensitive attributes.
    """

    CHURN_SEX_ATTRIBUTE_INDEX = 0

    dataset_orig = get_churn_data(path)

    continous_features = ['AGEC','POP','POI','PREMIUM','PAYCNT','RECNT','DELCNT']
    continous_features_indices = [dataset_orig.feature_names.index(feat) for feat in continous_features]

    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=True, seed=seed)


    SS = StandardScaler().fit(dataset_orig_train.features[:, continous_features_indices])
    dataset_orig_train.features[:, continous_features_indices] = SS.transform \
        (dataset_orig_train.features[:, continous_features_indices])
    dataset_orig_test.features[:, continous_features_indices] = SS.transform \
        (dataset_orig_test.features[:, continous_features_indices])

    X_train = dataset_orig_train.features
    X_test = dataset_orig_test.features

    y_train = dataset_orig_train.labels
    y_test = dataset_orig_test.labels

    X_val = X_train[:len(X_test)]
    y_val = y_train[:len(X_test)]
    X_train = X_train[len(X_test):]
    y_train = y_train[len(X_test):]

    if sensitive_attributes != "sex":
        raise ValueError("Specify proper protected feature for dataset 'churn'")
    else:
        # sex id = 0
        A_train = X_train[:, CHURN_SEX_ATTRIBUTE_INDEX]
        A_val = X_val[:, CHURN_SEX_ATTRIBUTE_INDEX]
        A_test = X_test[:, CHURN_SEX_ATTRIBUTE_INDEX]

    return X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test


def get_telecomkaggle_data(path, semi_synth_bias, bias_ratio):
    """
    Load and preprocess the TelecomKaggle dataset.

    Parameters:
    path (str): The path to the dataset.
    semi_synth_bias (str): The type of semi-synthetic bias to apply.
    bias_ratio (float): The ratio of bias to apply.

    Returns:
    BinaryLabelDataset: The preprocessed dataset.
    """
    headers = ['customerID','gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines',
               'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
               'StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges','Churn']

    data_types = {'tenure':float,
                  'MonthlyCharges':float,
                  'TotalCharges':float}

    df = pd.read_csv(os.path.join(path, "TelecomKaggle.csv"), header=0, sep=';', engine='python', dtype=data_types)
    df.columns = headers

    df.drop(['customerID'], axis=1, inplace=True)

    df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})
    df['gender'] = df['gender'].replace({'Female': 0, 'Male': 1})
    df['Partner'] = df['Partner'].replace({'No': 0, 'Yes': 1})
    df['Dependents'] = df['Dependents'].replace({'No': 0, 'Yes': 1})
    df['PhoneService'] = df['PhoneService'].replace({'No': 0, 'Yes': 1})
    df['MultipleLines'] = df['MultipleLines'].replace({'No': 0, 'No phone service': 0, 'Yes': 1})

    # Encode categorical features
    df = pd.get_dummies(df,
                        columns=['InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                 'PaymentMethod']).replace([False, True], [0, 1])

    df = df.dropna()

    if semi_synth_bias == 'random_flip':
        df = semi_synthetic_random_flip(df=df, bias_ratio=bias_ratio, sa='gender', target_label='Churn')

    elif semi_synth_bias == 'informed_flip':
        df = semi_synthetic_informed_flip(df=df, bias_ratio=bias_ratio, sa='gender', target_label='Churn')

    elif semi_synth_bias == 'none':
        pass # Do nothing

    else:
        raise ValueError("Specify proper semi-synthetic bias for dataset 'TelecomKaggle' ('random_flip', 'informed_flip', 'none')")

    return BinaryLabelDataset(df=df, label_names=['Churn'], protected_attribute_names=['gender'])


def preprocess_telecomkaggle_data(seed=0, path="", sensitive_attributes="sex", semi_synth_bias='none', bias_ratio=0.0, train_size=0.66):
    """
    Preprocess the TelecomKaggle dataset.

    Parameters:
    seed (int): Random seed for reproducibility.
    path (str): The path to the dataset.
    sensitive_attributes (str): The sensitive attribute to be used.
    semi_synth_bias (str): The type of semi-synthetic bias to apply.
    bias_ratio (float): The ratio of bias to apply.
    train_size (float): The proportion of the dataset to include in the train split.

    Returns:
    tuple: Training, validation, and test sets along with their labels and sensitive attributes.
    """

    TELECOMKAGGLE_SEX_ATTRIBUTE_INDEX = 0

    dataset_orig = get_telecomkaggle_data(path, semi_synth_bias, bias_ratio)

    continous_features = ['tenure','MonthlyCharges','TotalCharges']
    continous_features_indices = [dataset_orig.feature_names.index(feat) for feat in continous_features]

    dataset_orig_train, dataset_orig_test = dataset_orig.split([train_size], shuffle=True, seed=seed)

    SS = StandardScaler().fit(dataset_orig_train.features[:, continous_features_indices])
    dataset_orig_train.features[:, continous_features_indices] = SS.transform \
        (dataset_orig_train.features[:, continous_features_indices])
    dataset_orig_test.features[:, continous_features_indices] = SS.transform \
        (dataset_orig_test.features[:, continous_features_indices])

    X_train = dataset_orig_train.features
    X_test = dataset_orig_test.features

    y_train = dataset_orig_train.labels
    y_test = dataset_orig_test.labels

    X_val = X_train[:len(X_test)]
    y_val = y_train[:len(X_test)]
    X_train = X_train[len(X_test):]
    y_train = y_train[len(X_test):]

    if sensitive_attributes == 'sex':
        # sex id = 0
        A_train = np.expand_dims(X_train[:, TELECOMKAGGLE_SEX_ATTRIBUTE_INDEX], axis=1)
        A_val = np.expand_dims(X_val[:, TELECOMKAGGLE_SEX_ATTRIBUTE_INDEX], axis=1)
        A_test = np.expand_dims(X_test[:, TELECOMKAGGLE_SEX_ATTRIBUTE_INDEX], axis=1)

    else:
        raise ValueError("Specify proper protected feature for dataset 'TelecomKaggle' ('sex')")

    return X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test


# Custom dataset class that extends torch.utils.data.TensorDataset and converts input pandas DataFrames to tensors.
class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()


def semi_synthetic_random_flip(df, bias_ratio, sa, target_label):
    """
    Apply semi-synthetic random flip bias to the dataset.

    Parameters:
    df (pd.DataFrame): The dataset.
    bias_ratio (float): The ratio of bias to apply.
    sa (str): The sensitive attribute.
    target_label (str): The target label.

    Returns:
    pd.DataFrame: The biased dataset.
    """

    if not set(df[sa].unique()) == {0, 1} or not set(df[target_label].unique()) == {0, 1}:
        raise ValueError(f"SA ({sa}) and target_label ({target_label}) columns must have binary values (0 or 1).")

    # df contains X, y, A
    # Split the DataFrame into features (X) and target (y)
    X = df.drop(columns=[target_label, sa])
    y = df[target_label]
    A = df[sa]

    # Calculate the number of rows to change
    num_rows_to_change = int(bias_ratio * len(df[(A == 0) & (y == 0)]))

    # Randomly select rows to change
    rows_to_change = np.random.choice(df[(A == 0) & (y == 0)].index, size=num_rows_to_change, replace=False)

    # Update the 'Churn' values in the selected rows to 1
    df.loc[rows_to_change, target_label] = 1

    return df

def semi_synthetic_informed_flip(df, bias_ratio, sa, target_label):
    """
    Apply semi-synthetic informed flip bias to the dataset.

    Parameters:
    df (pd.DataFrame): The dataset.
    bias_ratio (float): The ratio of bias to apply.
    sa (str): The sensitive attribute.
    target_label (str): The target label.

    Returns:
    pd.DataFrame: The biased dataset.
    """

    if not set(df[sa].unique()) == {0, 1} or not set(df[target_label].unique()) == {0, 1}:
        raise ValueError(f"SA ({sa}) and target_label ({target_label}) columns must have binary values (0 or 1).")

    # df contains X, y, A
    # Split the DataFrame into features (X) and target (y)
    X = df.drop(columns=[target_label, sa])
    y = df[target_label]
    A = df[sa]

    # train a random forest to predict y from X
    rf = RandomForestClassifier()
    rf.fit(X, y)

    # Add predictions to the DataFrame. now you have a dataframe containing: X, y, A, y_pred
    df['y_pred'] = rf.predict_proba(X)[:, 1]

    # For A==1, keep data unchanged
    # For A==0 and y==0, rank according to y_pred and flip the label y to 1 for the top highest bias_ratio fraction
    mask = (A == 0) & (y == 0)
    subset = df[mask]
    num_to_flip = int(bias_ratio * len(subset))
    indices_to_flip = subset.nlargest(num_to_flip, 'y_pred').index
    df.loc[indices_to_flip, target_label] = 1

    # Drop the y_pred column before returning the DataFrame
    df.drop(columns=['y_pred'], inplace=True)

    return df

