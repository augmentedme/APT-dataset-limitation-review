# This code is for calculating performance with original feature for various ML algorithms'''

import numpy as np
import datetime
import operator
import os.path
from pathlib import Path
import random
import os
import csv
from csv import writer
import time
import sys
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import train_test_split


def findColWithZeroValues(file_name):
    # Read the DataFrame from the file
    df = pd.read_csv(file_name)

    # Initialize a list to store columns with only zero values
    zero_columns = []

    # Iterate through each column along with its index number
    for idx, column in enumerate(df.columns):
        # Check if the column contains numeric data
        if np.issubdtype(df[column].dtype, np.number):
            # Check if all values in the column are zero
            if (df[column] == 0).all():
                zero_columns.append((idx, column))

    # Print or use the list of columns
    print("Columns with only zero values:")
    for idx, column in zero_columns:
        print("Index:", idx, "Column:", column)


def clean_data(df):
    '''
    This function used for removing unnecessary features, duplicates and inf values.
    :param df: Dataset
    :return:
    '''
    df['Stage'] = df['Stage'].str.lower()
    # list of columns to drop (columns 12,70,71,72 are zero values columns)
    drop_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 70, 71, 72, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87] # total 24 columns
    # list of columns to keep
    keep_cols = list(set(range(len(df.columns))) - set(drop_cols))
    df = df.iloc[:, keep_cols]
    mask = df.duplicated()  # list of duplicate rows
    df = df.loc[~mask]  # Keep only non-duplicated values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def load_data(file_name_train, file_name_test):
    '''
    This function loads and preprocess a training and test csv files
    :param file_name_train: Training csv file
    :param file_name_test: Testing csv file
    :return: X_train, X_test, y_train, y_test, column_names
    '''
    file_path_train = Path(file_name_train)
    file_path_test = Path(file_name_test)

    if not file_path_train.exists():
        raise FileNotFoundError(f"The file with name {file_name_train} does not exist.")

    df_train = pd.read_csv(file_path_train, encoding="ISO-8859-1")
    df_test = pd.read_csv(file_path_test, encoding="ISO-8859-1")

    df_train = clean_data(df_train)
    df_test = clean_data(df_test)

    X_train = df_train.iloc[:, :-1].to_numpy()
    X_test = df_test.iloc[:, :-1].to_numpy()

    #Standerdizing the data using Standardscaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    column_names = df_train.columns[:-1].tolist()

    y_train_reshaped = df_train.iloc[:, -1].values.reshape(-1, 1)
    y_test_reshaped = df_test.iloc[:, -1].values.reshape(-1, 1)

    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    y_train = onehot_encoder.fit_transform(y_train_reshaped)
    y_test = onehot_encoder.transform(y_test_reshaped)

    return X_train_scaled, X_test_scaled, y_train, y_test, column_names #With normalization

def get_seed_val(run_num):
    '''
    This function returns seed value for 30 runs on the basis of run_num
    The run_num will be taken from command line argument. run_num will be started from 1 in sgegrid.
    '''
    seed_val = [332, 70, 82, 684, 319, 292, 351, 634, 653, 451, 429, 168, 272, 225, 410, 654, 454, 164, 520, 427, 405,
                602, 643, 296, 608, 199, 430, 504, 389, 306]
    return (seed_val[run_num - 1])

def append_list_as_row(file_name, list_of_elem):
    '''
    This function append the rows for the file file_name
    :param file_name: The csv file which rows need to be appended
    :param list_of_elem:  The items of the rows to append
    :return: appended rows
    '''
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def dt_classifier(X_train, X_test, y_train, y_test):
    # classifier = dt(random_state=101, splitter='random') #This part is for experimental purpose
    classifier = dt(random_state=101)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    bal_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    return bal_accuracy, cm

def svm_classifier(X_train, X_test, y_train, y_test):
    "This is the best setting we found for our case"
    classifier = SVC(random_state=101)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    bal_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    return bal_accuracy

def rf_classifier(X_train, X_test, y_train, y_test):
    classifier = rf(random_state=101)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    bal_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    return bal_accuracy

def gnb_classifier(X_train, X_test, y_train, y_test):
    classifier = gnb()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    bal_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    return bal_accuracy

def knn_classifier(X_train, X_test, y_train, y_test):
    # classifier = knn(n_neighbors=3, weights='distance')
    classifier = knn()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    bal_accuracy= balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    return bal_accuracy

def mlp_classifier(X_train, X_test, y_train, y_test):
    bal_accuracy_list = []

    for seed in range(1, 31):
        seed_int = get_seed_val(seed)
        # Create an MLPClassifier with a different random seed for each run
        classifier = mlp(random_state=seed_int)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        bal_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
        bal_accuracy_list.append(bal_accuracy)

    # Calculate the average balanced accuracy across all runs
    avg_bal_accuracy = np.mean(bal_accuracy_list)
    std_dev = np.std(bal_accuracy_list)

    return avg_bal_accuracy, std_dev

def csv_result(dt_acc, svm_acc, gnb_acc, rf_acc, knn_acc, mtgp_acc, stdev_mtgp):
    if os.path.exists('Bal_Acc.csv'):
        print()
    else:
        with open('Bal_Acc.csv', 'a+', newline='') as f:
            header = ['dt_acc', 'svm_acc', 'gnb_acc', 'rf_acc', 'knn_acc',
                      'mtgp_acc', 'stdev_mtgp']
            filewriter = csv.DictWriter(f, fieldnames=header)
            filewriter.writeheader()
    row_contents = [str(dt_acc), str(svm_acc), str(gnb_acc), str(rf_acc), str(knn_acc), str(mtgp_acc), str(stdev_mtgp)]
    # Append a list as new line to an old csv file
    append_list_as_row('Bal_Acc.csv', row_contents)

if __name__ == "__main__":
    start_time = datetime.datetime.now()

    # Data Pre-processing
    training_file_name = "./train.csv"
    testing_file_name = "./test.csv"

    X_train, X_test, y_train, y_test, column_names = load_data(training_file_name, testing_file_name)


    print("Calculating performance ...")

    dt_acc, cm_dt = dt_classifier(X_train, X_test, y_train, y_test)
    svm_acc = svm_classifier(X_train, X_test, y_train, y_test)
    gnb_acc = gnb_classifier(X_train, X_test, y_train, y_test)
    rf_acc= rf_classifier(X_train, X_test, y_train, y_test)
    knn_acc = knn_classifier(X_train, X_test, y_train, y_test)
    # mlp_acc, stdev_mlp = mlp_classifier(X_train, X_test, y_train, y_test)
    mtgp_acc, stdev_mtgp = MTGP_Classifier(X_train, X_test, y_train, y_test)
    print("Balanced Accuracy with existing feature Using DT: ", dt_acc)
    print("Confusion matrix with existing feature Using DT: ", cm_dt)

    # Generating CSV files to save balanced accuracy result
    csv_result(dt_acc, svm_acc, gnb_acc, rf_acc, knn_acc, mtgp_acc, stdev_mtgp)


    # Calculating the time for each run
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print("Total time spent: ", total_time)
