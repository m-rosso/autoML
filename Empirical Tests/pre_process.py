####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

__author__ = 'm-rosso'

import pandas as pd
import numpy as np
from copy import deepcopy

from utils import classify_variables, assessing_missings, missings_detection, data_consistency
from transformations import applying_log_transf, applying_standard_scale, applying_one_hot
from transformations import treating_missings

####################################################################################################################################
####################################################################################################################################
#############################################################FUNCTIONS##############################################################

####################################################################################################################################
# Function that applies distinct functions and classes in order to pre-process training and test data:

def pre_process(training_data, test_data, vars_to_drop, log_transform=True, standardize=True):
    """
    Function that applies distinct functions and classes in order to pre-process training and test data.
    
    The implemented procedures are log-transformation and standard scaling of numerical variables, missing values
    treatment, and one-hot encoding for transforming categorical variables.
    
    :param training_data: training data.
    :type training_data: dataframe.
    :param test_data: test data.
    :type test_data: dataframe.
    :param vars_to_drop: collection of variables that should not be considered during data pre-processing.
    :type vars_to_drop: list.
    :param log_transform: indicates whether to log-transform numerical variables.
    :type log_transform: boolean.
    :param standardize: indicates whether to standard scale numerical variables.
    :type standardize: boolean.
    
    :return: training and test data pre-processed.
    :rtype: tuple.
    """
    
    df_train = training_data.copy()
    df_test = test_data.copy()
    drop_vars = deepcopy(vars_to_drop)
    
    print('---------------------------------------------------------------------------------------------------------')
    print('\033[1mCLASSIFYING FEATURES AND EARLY SELECTION\033[0m')
    print('\n')
    
    class_variables = classify_variables(dataframe=df_train, vars_to_drop=drop_vars, test_data=df_test)
    
    # Lists of variables:
    cat_vars = class_variables['cat_vars']
    binary_vars = class_variables['binary_vars']
    cont_vars = class_variables['cont_vars']

    print('---------------------------------------------------------------------------------------------------------')
    print('\n')
    
    print('---------------------------------------------------------------------------------------------------------')
    print('\033[1mASSESSING MISSING VALUES\033[0m')
    print('\n')
    
    # Assessing missing values:
    print('\033[1mTraining data:\033[0m')
    missings_train = assessing_missings(dataframe=df_train)
    print('\n\033[1mTest data:\033[0m')
    missings_test = assessing_missings(dataframe=df_test)
    print('\n')
    
    print('---------------------------------------------------------------------------------------------------------')
    print('\n')
    
    print('---------------------------------------------------------------------------------------------------------')
    print('\033[1mAPPLYING LOGARITHMIC TRANSFORMATION OVER NUMERICAL DATA\033[0m')
    print('\n')

    # Variables that should not be log-transformed:
    not_log = [c for c in df_train.columns if c not in cont_vars]

    if log_transform:
        print('\033[1mTraining data:\033[0m')
        df_train = applying_log_transf(dataframe=df_train, not_log=not_log)

        print('\033[1mTest data:\033[0m')
        df_test = applying_log_transf(dataframe=df_test, not_log=not_log)
        print('\n')


    else:
        print('\033[1mNo transformation performed!\033[0m')
        print('\n')

    print('---------------------------------------------------------------------------------------------------------')
    print('\n')
    
    print('---------------------------------------------------------------------------------------------------------')
    print('\033[1mAPPLYING STANDARD SCALE TRANSFORMATION OVER NUMERICAL DATA\033[0m')
    print('\n')

    # Inputs that should not be standardized:
    not_stand = [c for c in df_train.columns if c.replace('L#', '') not in cont_vars]

    if standardize:
        scaled_data = applying_standard_scale(training_data=df_train, not_stand=not_stand,
                                              test_data=df_test)
        df_train_scaled = scaled_data['training_data']
        df_test_scaled = scaled_data['test_data']

    else:
        df_train_scaled = df_train.copy()
        df_test_scaled = df_test.copy()

        print('\033[1mNo transformation performed!\033[0m')

    print('\n')
    print('---------------------------------------------------------------------------------------------------------')
    print('\n')
    
    print('---------------------------------------------------------------------------------------------------------')
    print('\033[1mTREATING MISSING VALUES\033[0m')
    print('\n')

    print('\033[1mTreating missing values of training data...\033[0m')
    df_train_scaled = treating_missings(dataframe=df_train_scaled, cat_vars=cat_vars,
                                        drop_vars=drop_vars)

    print('\033[1mTreating missing values of test data...\033[0m')
    df_test_scaled = treating_missings(dataframe=df_test_scaled, cat_vars=cat_vars,
                                       drop_vars=drop_vars)

    print('\n')
    print('---------------------------------------------------------------------------------------------------------')
    print('\n')
    
    print('---------------------------------------------------------------------------------------------------------')
    print('\033[1mTRANSFORMING CATEGORICAL FEATURES\033[0m')
    print('\n')
    
    transf_data = applying_one_hot(df_train_scaled, cat_vars, test_data=df_test_scaled)
    df_train_scaled = transf_data['training_data']
    df_test_scaled = transf_data['test_data']

    print(f'\033[1mShape of df_train_scaled:\033[0m {df_train_scaled.shape}.')
    print(f'\033[1mShape of df_test_scaled:\033[0m {df_test_scaled.shape}.')
    print('\n')
    
    print('---------------------------------------------------------------------------------------------------------')
    print('\n')
    
    print('---------------------------------------------------------------------------------------------------------')
    print('\033[1mFINAL ASSESSMENT OF MISSINGS AND CHECKING DATASETS CONSISTENCY\033[0m')
    print('\n')
    
    # Assessing missing values (training data):
    missings_detection(df_train_scaled, name=f'df_train_scaled')

    # Assessing missing values (test data):
    missings_detection(df_test_scaled, name=f'df_test_scaled')
    
    # Checking datasets structure:
    df_test_scaled = data_consistency(dataframe=df_train_scaled,
                                      test_data=df_test_scaled)['test_data']
    
    print('---------------------------------------------------------------------------------------------------------')
    print('\n')
    
    return df_train, df_test, df_train_scaled, df_test_scaled
