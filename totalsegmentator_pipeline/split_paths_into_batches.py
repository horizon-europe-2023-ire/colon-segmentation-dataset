"""
===============================================================================
 Script: split_paths_into_batches.py
 Purpose:
     Divide a list of image file paths into smaller batch files for parallel
     or incremental processing.

 Description:
     - Reads a text file (e.g. image_paths.txt) containing full paths to image
       volumes (e.g. .mha.gz files).
     - Splits the list into chunks of a specified batch size.
     - Writes each chunk to a new text file inside an output directory
       (e.g. batch_paths/), named sequentially as batch_1.txt, batch_2.txt, etc.

 Example:
     Input:
         image_paths.txt:
             ../data/converted/sub001/scan1.mha.gz
             ../data/converted/sub002/scan2.mha.gz
             ../data/converted/sub003/scan3.mha.gz
             ../data/converted/sub004/scan4.mha.gz

     Parameters:
         batch_size = 2
         output_dir = "batch_paths"

     Output files:
         batch_paths/batch_1.txt  → contains first 2 paths
         batch_paths/batch_2.txt  → contains remaining 2 paths

 Parameters:
     image_paths_file : str
         Path to the input text file listing all image paths.
     output_dir : str
         Directory where batch text files will be saved.
     batch_size : int
         Number of image paths per batch file.

 Notes:
     - The output directory is created automatically if it does not exist.
     - Each batch file contains one image path per line.
     - Useful for distributing workload across multiple machines or jobs.
===============================================================================
"""

import os

def split_paths_into_batches(image_paths_file, output_dir, batch_size):
    with open(image_paths_file, 'r') as f:
        image_paths = f.read().splitlines()

    # Split image paths into batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_file = f"{output_dir}/batch_{i // batch_size + 1}.txt"

        with open(batch_file, 'w') as bf:
            bf.write("\n".join(batch_paths))

if __name__ == "__main__":
    image_paths_file = "image_paths.txt"
    output_dir = "batch_paths"
    batch_size = 10  # Adjust batch size as needed

    os.makedirs(output_dir, exist_ok=True)
    split_paths_into_batches(image_paths_file, output_dir, batch_size)
