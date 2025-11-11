"""
===============================================================================
 Script: generate_random_colon_slices.py
 Purpose:
     Generate a dataset of random 2D PNG slices from 3D medical scans (.mha)
     for use with RootPainter — to train where the fluid/colon is
     located in the scans.

 Description:
     - For each .mha volume, the script selects a number of random axial slices.
     - It skips uniform slices (where all voxel values are identical).
     - Each valid slice is normalized to [0, 255], resized to 1000×1000 pixels,
       and saved as a PNG file in the specified output directory.
     - The resulting dataset can be used directly with RootPainter for
       annotation or inference.

 Usage:
     python generate_random_colon_slices.py <input_dir> <output_dir> <num_slices_per_volume> [max_files]

 Example:
     python generate_random_colon_slices.py ../data/converted ../data/rootpainter_dataset 5 10

 Arguments:
     <input_dir>             Directory containing .mha files.
     <output_dir>            Directory to save the PNG slices.
     <num_slices_per_volume> Number of random slices to extract from each volume.
     [max_files]             (Optional) Maximum number of .mha files to process.

 Notes:
     - Images are saved as "<basename>_random_slice###.png".
     - If a volume contains many uniform slices, fewer slices may be saved.
     - The output directory is created automatically if it doesn’t exist.
===============================================================================
"""

import SimpleITK as sitk
import numpy as np
import os
import random
from PIL import Image
import sys


def extract_and_save_random_slices_from_file(mha_file_path, output_dir, num_slices):
    """
    Extracts random slices in the axial plane from a .mha file and saves them as PNG files.
    Skips uniform slices and ensures non-duplicate random sampling.
    """
    # Load the .mha file
    image = sitk.ReadImage(mha_file_path)

    # Get the 3D image array
    image_array = sitk.GetArrayFromImage(image)

    # Extract the base name for the file without extension
    base_name = os.path.basename(mha_file_path).split(".mha")[0]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    total_slices = image_array.shape[0]
    selected_slices = set()
    saved_slices = 0
    attempts = 0
    max_attempts = 10 * num_slices  # Avoid infinite loops

    while saved_slices < num_slices and attempts < max_attempts:
        attempts += 1

        # Random slice index
        slice_index = random.randint(0, total_slices - 1)

        # Skip duplicates
        if slice_index in selected_slices:
            continue

        slice_image = image_array[slice_index, :, :]

        # Skip uniform slices
        if np.min(slice_image) == np.max(slice_image):
            continue

        # Normalize to [0, 255]
        slice_image = ((slice_image - np.min(slice_image)) / (np.max(slice_image) - np.min(slice_image)) * 255).astype(np.uint8)

        # Convert to PIL and resize
        pil_image = Image.fromarray(slice_image)
        pil_image = pil_image.resize((1000, 1000), Image.Resampling.LANCZOS)

        # Save as PNG
        output_file = os.path.join(output_dir, f"{base_name}_random_slice{saved_slices+1:03d}.png")
        pil_image.save(output_file, "PNG")

        print(f"Saved slice {slice_index} as {output_file}")

        selected_slices.add(slice_index)
        saved_slices += 1

    if saved_slices < num_slices:
        print(f"⚠️  Only {saved_slices}/{num_slices} slices saved for {mha_file_path} (uniform or duplicate slices).")


def extract_and_save_random_slices_from_directory(input_dir, output_dir, num_slices, max_files=None):
    """
    Extract random slices from a limited number of .mha files in a directory and its subdirectories.
    """
    file_count = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mha"):
                if max_files is not None and file_count >= max_files:
                    print("Reached the maximum number of files to process.")
                    return
                mha_file_path = os.path.join(root, file)
                print(f"Processing file: {mha_file_path}")
                extract_and_save_random_slices_from_file(mha_file_path, output_dir, num_slices)
                file_count += 1


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python generate_random_colon_slices.py <input_dir> <output_dir> <num_slices_per_volume> [max_files]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    num_slices = int(sys.argv[3])
    max_files = int(sys.argv[4]) if len(sys.argv) > 4 else None

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Slices per volume: {num_slices}")
    print(f"Max files to process: {max_files if max_files else 'All'}")
    print("Starting slice extraction...")

    extract_and_save_random_slices_from_directory(input_dir, output_dir, num_slices, max_files)

    print("✅ Dataset generation complete.")