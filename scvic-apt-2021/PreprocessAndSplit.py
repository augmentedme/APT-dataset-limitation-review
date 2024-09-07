''' The Training and Tesing csv files are downloaded from the original dataset paper mentioned in the paper.
or search for SCVIC-APT-2021 in IEEE dataport
'''

import csv
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pathlib import Path

# Specify the input directory and output file
raw_file1 = "Training.csv"
raw_file2 = "Testing.csv"
output_file_path = "combined.csv"

train_file_path = "train.csv"
test_file_path = "test.csv"

def combine_csv():
    # Check if both files exist
    if os.path.exists(raw_file1) and os.path.exists(raw_file2):
        # Read both CSV files into DataFrames
        df_a = pd.read_csv(raw_file1)
        df_b = pd.read_csv(raw_file1)
        # Concatenate DataFrames along rows (assuming they have the same columns)
        df_concat = pd.concat([df_a, df_b], ignore_index=True)
        print("Printing data before preprocessing...")
        print(f"Shape of concatenated dataframe: {df_concat.shape}")
        if 'Label' in df_concat.columns:
            class_counts = df_concat['Label'].value_counts()
            print("\nNumber of instances per class:")
            print(class_counts)
        else:
            print("\nClass column not found. Available columns:")
            print(df_concat.columns)
        return df_concat

    else:
        print("One or both CSV files do not exist.")

def abnormal_row_remove():
    #combining train and test and calling the df here.
    combined_df = combine_csv()
    # Create an empty list to store all valid lines from selected files
    all_valid_rows = []
    # Counter to keep track of lines removed
    lines_removed_count = 0

    # Get the number of columns from the first row (header) or assume all rows have the same number of columns
    num_columns = len(combined_df.columns)
    # Append the header as the first valid row
    all_valid_rows.append(combined_df.columns.tolist())

    # Example of checking and appending valid rows
    for index, row in combined_df.iterrows():
        # Check if the number of values in the row is equal to the number of columns
        if len(row) == num_columns:
            all_valid_rows.append(row)
        else:
            lines_removed_count += 1  # Count removed rows where number of values doesn't match num_columns

    return all_valid_rows, lines_removed_count

def create_combined_csv(all_valid_rows):
    # Open the output csv file in write mode
    with open(output_file_path, "w", newline="") as output_file:
        # Create a csv writer object
        writer = csv.writer(output_file)
        # Write the valid lines to the output file
        writer.writerows(all_valid_rows)

def clean_data(df):
    '''
    This function used for removing unnecessary features, duplicates and inf values.
    :param df: Dataset
    :return: df
    '''
    # Replace spaces and slashes (\) with underscores (_) in column names
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('/', '_')
    # list of columns to drop.
    drop_cols = ['Flow_ID', 'Src_IP', 'Src_Port', 'Dst_IP', 'Timestamp']
    # list of columns to keep
    # keep_cols = list(set(range(len(df.columns))) - set(drop_cols))
    keep_cols = list(set(df.columns) - set(drop_cols))
    # Select only the columns to keep
    df = df[keep_cols]
    # Identify and count duplicate rows
    duplicate_count = df.duplicated().sum()
    # Count the number of duplicate rows
    print(f"Number of duplicate rows: {duplicate_count}")
    # Identify duplicate rows in the dataframe
    mask = df.duplicated()
    # Keep only the non-duplicated rows by using the inverted mask
    df = df.loc[~mask]
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Count the number of rows with NaN values
    num_nan_rows = df.isna().any(axis=1).sum()
    print(f"Number of rows with NaN values: {num_nan_rows}")
    # Drop rows with any NaN values
    df.dropna(inplace=True)

    # Move 'Label' column to the end
    label_col = df.pop('Label')
    df['Label'] = label_col

    # Replace negative values with zero in all columns except the 'Label' column
    df.iloc[:, :-1] = df.iloc[:, :-1].where(df.iloc[:, :-1] >= 0, 0)  # Efficient vectorized approach
    # Identify columns with only zero values
    zero_value_cols = df.columns[(df == 0).all()]
    # Print the number of columns with only zero values and their names
    print(f"Number of columns with only zero values: {len(zero_value_cols)}")
    if len(zero_value_cols) > 0:
        print("Columns with only zero values:", zero_value_cols.tolist())
    # Drop columns with only zeros.
    df = df.loc[:, (df != 0).any(axis=0)]
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
    # print("\nDataFrame Info:")
    # df.info()

    df['Label'] = df['Label'].str.lower()

    df = clean_data(df)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_value_counts = y.value_counts()
    print(f"Number of instances per class in comined_csv: {X_value_counts}")

    print(f"The shape of X after processing: {X.shape}")

    column_names = df.columns[:-1].tolist()

    # Split the data into training and testing sets (80-20 split) considering Stage column as label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)

    return X_train, X_test, y_train, y_test, column_names

#Cleaning abnormal rows
all_valid_rows, lines_removed_count = abnormal_row_remove()
# Combining train and test csvs
create_combined_csv(all_valid_rows)

print(f"Lines removed due to more columns than the header: {lines_removed_count}")
print(f"Total valid lines remain: {len(all_valid_rows)}")

#Data Pre-processing
X_train, X_test, y_train, y_test, column_names = load_data(output_file_path)
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


