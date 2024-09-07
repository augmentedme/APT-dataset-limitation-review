import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


# Input and output file paths
input_file_path = "output.csv"
train_file_path = "train.csv"
test_file_path = "test.csv"

def clean_data(df):
    '''
    This function used for removing unnecessary features, duplicates and inf values.
    :param df: Dataset
    :return:
    '''
    df['Stage'] = df['Stage'].str.lower()
    # list of columns to drop (columns 12,70,71,72 are zero values columns)
    drop_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 70, 71, 72, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88] # total 24 columns
    # list of columns to keep
    keep_cols = list(set(range(len(df.columns))) - set(drop_cols))
    # Select only the columns to keep
    df = df.iloc[:, keep_cols]
    # Identify duplicate rows in the dataframe
    mask = df.duplicated()
    # Count the number of duplicate rows
    print(f"Number of duplicate rows: {mask.sum()}")
    # Keep only the non-duplicated rows by using the inverted mask
    df = df.loc[~mask]
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_nan_rows = df.isna().any(axis=1).sum()
    # Print the number of rows with NaN values
    print(f"Number of rows with NaN values: {num_nan_rows}")
    # Drop rows with any NaN values
    df.dropna(inplace=True)
    return df

def load_data(file_path):
    '''
    This function loads and preprocess a training and test csv files
    :param file_path: csv file
    :return: X_train, X_test, y_train, y_test, column_names
    '''
    try:
        df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None


    df = clean_data(df)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    nan_values_X = X.isna().any(axis=1).sum()

    print(f"The shape of X after processing: {X.shape}")
    print(f"Total NaN values in cleaned X: {nan_values_X}")

    column_names = df.columns[:-1].tolist()

    # Split the data into training and testing sets (80-20 split) considering Stage column as label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    return X_train, X_test, y_train, y_test, column_names


#Data Pre-processing
X_train, X_test, y_train, y_test, column_names = load_data(input_file_path)

print(f"Column names are: {column_names}")


# Save the training data to a new CSV file
train_data = pd.concat([X_train, y_train], axis=1)  # Combine features and target variable
train_data.to_csv(train_file_path, index=False)
print(f"Training data saved as {train_file_path}")

# Save the testing data to a new CSV file
test_data = pd.concat([X_test, y_test], axis=1)  # Combine features and target variable
test_data.to_csv(test_file_path, index=False)
print(f"Testing data saved as {test_file_path}")


# Printing shape of train and test data
print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

X_train_value_counts = y_train.value_counts()
print(f"Number of instances per class in y_train: {X_train_value_counts}")
X_test_value_counts = y_test.value_counts()
print(f"Number of instances per class in y_test: {X_test_value_counts}")
