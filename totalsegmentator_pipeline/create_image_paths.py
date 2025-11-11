"""
===============================================================================
 Script: create_image_paths_file.py
 Purpose:
     Generate a text file listing the full paths of all converted CT colonography
     image volumes (.mha.gz or .mha.zip) in a given directory tree.

 Description:
     - This script recursively scans an input folder (e.g. ../data/converted/)
       for all files ending with `.mha.gz` or `.mha.zip`.
     - For each matching file, it writes its absolute (or relative) path as a
       new line into an output text file (e.g. image_paths.txt).
     - The resulting text file can be used as an index or manifest for batch
       processing pipelines such as segmentation or filtering scripts.

 Typical usage:
     python create_image_paths_file.py

 Example:
     If the directory structure is:
         ../data/converted/
             sub001/scan1.mha.gz
             sub002/scan1.mha.zip
     Then the output file image_paths.txt will contain:
         ../data/converted/sub001/scan1.mha.gz
         ../data/converted/sub002/scan1.mha.zip

 Parameters:
     input_dir   : str
         Root folder where converted image files are stored.
     output_txt  : str
         Output text file to store the full image paths.

===============================================================================
"""

import os

def create_image_paths_file(input_dir, output_txt):
    with open(output_txt, 'w') as f:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".mha.gz") or file.endswith(".mha.zip"):
                    full_path = os.path.join(root, file)
                    f.write(full_path + "\n")

if __name__ == "__main__":
    input_dir = "../data/converted"  # Input directory with converted images
    output_txt = "image_paths.txt"   # Output text file to store the paths

    create_image_paths_file(input_dir, output_txt)
