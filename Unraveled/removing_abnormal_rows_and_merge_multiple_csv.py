''' This script will remove the lines that has columns more than the header line, from all CSVs in a specified range and then merge all of CSVs into and output.csv file. The header will be added once.
Downlaod all the csv files for network flows from Unraveled gitlab link and rename those file from 1 to 173
Link : https://gitlab.com/asu22/unraveled/-/tree/master/data/network-flows?ref_type=heads '''


import csv
import os

# Specify the input directory and output file
input_directory = "./"  # to specify the current directory
output_file_path = "output.csv"  # change the name as necessary


start_file_number = 1
end_file_number = 173

# Create an empty list to store all valid lines from selected files
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

# Open the output csv file in write mode
with open(output_file_path, "w", newline="") as output_file:
    # Create a csv writer object
    writer = csv.writer(output_file)
    # Write the valid lines to the output file
    writer.writerows(all_valid_lines)

print(f"Merged CSV files {start_file_number} to {end_file_number} into {output_file_path}")
print(f"Lines removed due to more columns than the header: {lines_removed_count}")
print(f"Total valid lines remain: {len(all_valid_lines)}")
