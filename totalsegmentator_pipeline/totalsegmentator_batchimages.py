import os
import multiprocessing
import subprocess
from pathlib import Path

def run_totalsegmentator(input_image_path):
    # Get the corresponding output path by replacing the parent directory
    relative_path = os.path.relpath(input_image_path, "/home/smp884/IRE/data/CT/converted")
    output_image_path = os.path.join("/home/smp884/IRE/data/CT/segmentations_totalsegmentator", relative_path.replace(".mha.gz", "").replace(".mha.zip", ""))

    # Ensure the output directory exists
    os.makedirs(output_image_path, exist_ok=True)

    print("Input: ", input_image_path, " Output: ", output_image_path)

    # Run the totalsegmentator_oneimage.py script for each image
    subprocess.run([
        "python", "/home/smp884/IRE/totalsegmentator_oneimage.py", input_image_path, output_image_path
    ], stderr=open('/home/smp884/IRE/warnings.log', 'a'))

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
