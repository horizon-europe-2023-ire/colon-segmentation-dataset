"""
===============================================================================
 Script: totalsegmentator_batchimages.py
 Purpose:
     Automate batch segmentation of multiple medical image volumes using
     TotalSegmentator by processing all image paths listed in a text file.

 Description:
     - Reads a text file containing paths to .mha.gz or .mha.zip images.
     - For each image, constructs a corresponding output directory under
       ../data/segmentations_totalsegmentator/ that mirrors the structure
       of ../data/converted/.
     - Executes the `totalsegmentator_oneimage.py` script separately for each
       input image to perform segmentation.
     - Uses multiprocessing to process multiple images in parallel, improving
       efficiency for large datasets.
     - All warnings from subprocess executions are appended to `warnings.log`.


 Usage:
     python totalsegmentator_batchimages.py <image_paths_file>

 Example:
     Input:
         image_paths_file = "batch_paths/batch_1.txt"
         (each line: ../data/converted/sub001/scan1.mha.gz, etc.)

     Command:
         python totalsegmentator_batchimages.py batch_paths/batch_1.txt

     Output:
         ../data/segmentations_totalsegmentator/sub001/scan1/
         ../data/segmentations_totalsegmentator/sub002/scan2/
         ...

 Parameters:
     image_paths_file : str
         Path to a text file containing full paths to input images.
     input_image_path : str
         (internal) Path to a single image passed to TotalSegmentator.
     output_image_path : str
         (internal) Directory where segmentation results will be saved.

 Notes:
     - The script assumes `totalsegmentator_oneimage.py` is available and
       properly configured in the same working environment.
     - Output folder structure mirrors that of the converted data.
     - Multiprocessing speeds up batch processing but can be adjusted by
       modifying the `processes` argument in the Pool call.
     - All warnings are logged in `warnings.log` for later inspection.
===============================================================================
"""

import os
import multiprocessing
import subprocess
from pathlib import Path

def run_totalsegmentator(input_image_path):
    # Get the corresponding output path by replacing the parent directory
    relative_path = os.path.relpath(input_image_path, "../data/converted")
    output_image_path = os.path.join("../data/segmentations_totalsegmentator", relative_path.replace(".mha.gz", "").replace(".mha.zip", ""))

    # Ensure the output directory exists
    os.makedirs(output_image_path, exist_ok=True)

    print("Input: ", input_image_path, " Output: ", output_image_path)

    # Run the totalsegmentator_oneimage.py script for each image
    subprocess.run([
        "python", "totalsegmentator_oneimage.py", input_image_path, output_image_path
    ], stderr=open('warnings.log', 'a'))

def process_batch(image_paths_file):
    with open(image_paths_file, 'r') as f:
        image_paths = f.read().splitlines()

    # Use multiprocessing to process multiple images in parallel
    with multiprocessing.Pool(processes=4) as pool:
        pool.map(run_totalsegmentator, image_paths)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python totalsegmentator_batchimages.py <image_paths_file>")
        sys.exit(1)

    image_paths_file = sys.argv[1]
    process_batch(image_paths_file)


