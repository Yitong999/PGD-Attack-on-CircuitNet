import pandas as pd

def filter_csv(input_file_path, output_file_path):
    # Load the CSV file
    data = pd.read_csv(input_file_path)

    # Filter out rows where ClassId is outside the range 0 to 9
    filtered_data = data[data['ClassId'].between(0, 9)]

    # Save the filtered data to a new CSV file
    filtered_data.to_csv(output_file_path, index=False)
    print(f"Filtered data saved to {output_file_path}")

if __name__ == "__main__":
    # Define the input and output file paths
    input_file_path = 'German/Train.csv' 
    output_file_path = 'German/Train_filtered.csv' 


    # Call the function with the file paths
    filter_csv(input_file_path, output_file_path)

    input_file_path = 'German/Test.csv' 
    output_file_path = 'German/Test_filtered.csv' 

    # Call the function with the file paths
    filter_csv(input_file_path, output_file_path)
