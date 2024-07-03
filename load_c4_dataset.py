import os 
import subprocess
import gzip
import shutil

def load_c4_subset(iterations: int = 5):
    directory = 'dataset/c4/en/'
    base_command = 'git lfs pull --include "en/c4-train.{:05d}-of-01024.json.gz"'
    for i in range(0, iterations):
        command = base_command.format(i)
        # Print the command for debugging purposes
        print(f"Executing: {command}")
    
        # Execute the command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
        # Print the result for debugging purposes
        if result.returncode == 0:
            print(f"Successfully executed: {command}")
            
            # Define the filename
            filename = f'c4-train.{i:05d}-of-01024.json.gz'
            gz_file_path = os.path.join(directory, filename)
            json_file_path = os.path.join(directory, filename[:-3])
            
            # Check if the .gz file exists
            if os.path.exists(gz_file_path):
                print(f"Unpacking: {gz_file_path}")
                
                # Unpack the .gz file
                with gzip.open(gz_file_path, 'rb') as f_in:
                    with open(json_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                        
                print(f"Unpacked to: {json_file_path}")
                
                # Delete the .gz file after unpacking
                os.remove(gz_file_path)
                print(f"Deleted: {gz_file_path}")
            else:
                print(f"File does not exist: {gz_file_path}")
        else:
            print(f"Error executing: {command}")
            print(result.stderr)
        
 
 
if __name__ == '__main__':
    #Each File is 309 MB
    #Imagine we have 100 GB of Storage so we can do 100 * 1000 / 309 = 323 Files 
    storage = 100_000 #MB
    number_of_files = int(storage / 309)
    load_c4_subset(5)
    