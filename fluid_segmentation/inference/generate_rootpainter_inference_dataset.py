"""
===============================================================================
 Script: generate_rootpainter_inference_dataset.py
 Purpose:
     This script prepares a dataset of PNG images from 3D medical scans (.mha)
     for use with RootPainter — an annotation and segmentation tool. The PNGs
     can be used for inferring or labeling regions such as fluid within CT scans.

 Description:
     - The script loads 3D medical images stored in .mha format (MetaImage).
     - It extracts every axial (transverse) slice from each 3D volume.
     - Each slice is normalized to 8-bit grayscale [0–255], resized to 1000×1000
       pixels, and saved as an individual PNG image.
     - The resulting PNG images form a 2D dataset suitable for RootPainter input
       or inference.

 Usage:
     python generate_rootpainter_inference_dataset.py <input_dir> <output_dir> <max_files>

     Example:
         python generate_rootpainter_inference_dataset.py \
             ../data/converted ../data/rootpainter_input 5

 Arguments:
     <input_dir>   Path to the folder containing .mha files.
     <output_dir>  Destination folder for saving PNG slices.
     <max_files>   Maximum number of .mha volumes to process.

 Notes:
     - Each .mha file is saved as multiple PNG slices named:
         <basename>_slice000.png, <basename>_slice001.png, ...
     - All images are resized to 1000×1000 using high-quality LANCZOS resampling.
     - The script can be used to generate a small subset or the full dataset,
       depending on the 'max_files' parameter.

===============================================================================
"""

import SimpleITK as sitk
import numpy as np
import os
from PIL import Image
from pathlib import Path

def convert_mha_to_png(mha_file_path, output_dir):
    """
    Extracts all slices in the axial plane from a .mha file and saves them as PNG files.

    Parameters:
        mha_file_path (str): Path to the .mha file.
        output_dir (str): Directory to save the extracted slices as PNGs.
    """
    # Load the .mha file
    image = sitk.ReadImage(mha_file_path)

    # Get the 3D image array
    image_array = sitk.GetArrayFromImage(image)

    # Extract the base name for the file without extension
    base_name = os.path.basename(mha_file_path).split(".mha")[0]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all slices
    total_slices = image_array.shape[0]
    for slice_index in range(total_slices):
        slice_image = image_array[slice_index, :, :]

        # Normalize the slice to the range [0, 255]
        slice_image = ((slice_image - np.min(slice_image)) / (np.max(slice_image) - np.min(slice_image)) * 255).astype(np.uint8)

        # Convert to a PIL Image
        pil_image = Image.fromarray(slice_image)

        # Resize image to 1000 x 1000
        pil_image = pil_image.resize((1000, 1000), Image.Resampling.LANCZOS)

        # Save as PNG
        output_file = os.path.join(output_dir, f"{base_name}_slice{slice_index:03d}.png")
        pil_image.save(output_file, "PNG")

        print(f"Saved slice {slice_index} as {output_file}")

def process_directory(input_dir, output_dir, max_files=3):
    """
    Process a limited number of .mha files in a directory and save their slices as PNG images.

    Parameters:
        input_dir (str): Directory containing .mha files.
        output_dir (str): Directory to save the extracted slices.
        max_files (int): Maximum number of .mha files to process.
    """
    file_count = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mha"):
                if file_count >= max_files:
                    print(f"Reached the maximum limit of {max_files} files.")
                    return
                mha_file_path = os.path.join(root, file)
                print(f"Processing file: {mha_file_path}")
                convert_mha_to_png(mha_file_path, output_dir)
                file_count += 1
              
# Usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python generate_rootpainter_inference_dataset.py <input directory> <output directory> <max files to process>")
        sys.exit(1)

    input_dir = sys.argv[1] #Update input dir
    output_dir = sys.argv[2] #Update output dir dir
    max_files = int(sys.argv[3]) # Process max file to process

    print("Input directory:",input_dir)
    print("Output directory:",output_dir)
    print("Max files to process:",max_files)
    print("Starting the processing")
    process_directory(input_dir, output_dir, max_files=max_files)