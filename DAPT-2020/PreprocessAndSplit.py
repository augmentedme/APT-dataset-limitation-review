'''Copy all csv starting with prefix "custom_" from data folder of original dataset folder stored at https://gitlab.com/asu22/dapt2020.
Store all csv in a new folder sequencially with numerical naming such as 1,2,3,....

'''

import csv
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pathlib import Path

# Specify the input directory and output file
input_directory = "./"  # to specify the current directory
output_file_path = "output.csv"  # change the name as necessary
train_file_path = "train.csv"
test_file_path = "test.csv"

# Specify the range of CSV file numbers to merge. This is the range we will use to combine multiple CSVs with
# specified pattern in the name
start_file_number = 1
end_file_number = 25


def abnormal_row_remove():
    all_valid_lines = []
    # Counter to keep track of lines removed
    lines_removed_count = 0
    # Boolean variable to track whether the header has been written
    header_written = False
    # Loop through the specified range of CSV file numbers
    for file_number in range(start_file_number, end_file_number + 1):
        csv_file = os.path.join(input_directory, f"{file_number}.csv")
        # Check if the file exists
        if os.path.exists(csv_file):
            with open(csv_file, "r") as input_file:
                # Create a csv reader object
                reader = csv.reader(input_file)
                # Read the header line and store it in a variable
                header = next(reader)
                # Get the number of columns in the header line
                num_columns = len(header) if header is not None else None
                # Create an empty list to store the valid lines for the current file
                valid_lines = []
                # Append the header line to the valid lines list only if it hasn't been written yet
                if header is not None and not header_written:
                    valid_lines.append(header)
                    header_written = True

                # Loop through the rest of the lines in the current file
                for line in reader:
                    # Check if the number of columns in the line is equal to the header line
                    if num_columns is None or len(line) == num_columns:
                        # If yes, append the line to the valid lines list
                        valid_lines.append(line)
                    else:
                        lines_removed_count += 1  # Increment the lines_removed_count for each invalid line
                # Append the valid lines from the current file to the overall list
                all_valid_lines.extend(valid_lines)
        else:
            print(f"File {csv_file} not found.")

    return all_valid_lines, lines_removed_count

def create_combined_csv():
    # Open the output csv file in write mode
    with open(output_file_path, "w", newline="") as output_file:
        # Create a csv writer object
        writer = csv.writer(output_file)
        # Write the valid lines to the output file
        writer.writerows(all_valid_lines)

def clean_data(df):
    '''
    This function used for removing unnecessary features, duplicates and inf values.
    :param df: Dataset
    :return: df
    '''
    # Replace '/' with '_' in column names
    df.columns = df.columns.str.replace('/', '_')

    df['Stage'] = df['Stage'].str.lower()
    # list of columns to drop. The feature removes are fllow_id, src_ip, src_port, dst_ip, timestamp, activity
    drop_cols = [0, 1, 2, 3, 6, 83]
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
    # Replace negative values with zero (new addition). Doing these on all column except label column
    df.iloc[:, :-1] = df.iloc[:, :-1].where(df.iloc[:, :-1] >= 0, 0)  # Efficient vectorized approach
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

    df = clean_data(df)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print(f"The shape of X after processing: {X.shape}")

    column_names = df.columns[:-1].tolist()

    # Split the data into training and testing sets (80-20 split) considering Stage column as label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


    return X_train, X_test, y_train, y_test, column_names

#Cleaning abnormal rows
all_valid_lines, lines_removed_count = abnormal_row_remove()

#Combing all CSVs
create_combined_csv()
print(f"Merged CSV files {start_file_number} to {end_file_number} into {output_file_path}")
print(f"Lines removed due to more columns than the header: {lines_removed_count}")
print(f"Total valid lines remain: {len(all_valid_lines)}")

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

