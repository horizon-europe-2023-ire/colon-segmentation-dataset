import SimpleITK as sitk
import numpy as np
import os
import random
from PIL import Image

"""
Generate a datset for rootpainter extracting a number of slices randomly where the colon is prensent from each image and saving them intoa  different folder as .png
"""

def extract_and_save_random_slices_from_file(mha_file_path, output_dir, num_slices):
    """
    Extracts random slices in the axial plane from a .mha file and saves them as PNG files.

    Parameters:
        mha_file_path (str): Path to the .mha file.
        output_dir (str): Directory to save the extracted slices as JPEGs.
        num_slices (int): Number of random slices to extract.
    """
    # Load the .mha file
    image = sitk.ReadImage(mha_file_path)

    # Get the 3D image array
    image_array = sitk.GetArrayFromImage(image)

    # Extract the base name for the file without extension
    base_name = os.path.basename(mha_file_path).split(".mha")[0]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the total number of slices
    total_slices = image_array.shape[0]

    # Keep track of selected slices
    selected_slices = set()
    saved_slices = 0
    attempts = 0
    max_attempts = 10 * num_slices  # Avoid infinite loops in case of mostly uniform slices

    while saved_slices < num_slices and attempts < max_attempts:
        attempts += 1

        # Select a random slice index
        slice_index = random.randint(0, total_slices - 1)

        # Skip if this slice index has already been selected
        if slice_index in selected_slices:
            continue

        slice_image = image_array[slice_index, :, :]

        # Check if the slice is uniform
        if np.min(slice_image) == np.max(slice_image):
            # Skip this slice if it is of one color
            continue

        # Normalize the slice to the range [0, 255]
        slice_image = ((slice_image - np.min(slice_image)) / (np.max(slice_image) - np.min(slice_image)) * 255).astype(np.uint8)

        # Convert to a PIL Image
        pil_image = Image.fromarray(slice_image)

        # Resize image to 1000 x 1000
        pil_image = pil_image.resize((1000, 1000), Image.Resampling.LANCZOS)

        # Save as PNG
        output_file = os.path.join(output_dir, f"{base_name}_random_slice{saved_slices+1:03d}.png")
        pil_image.save(output_file, "PNG")

        print(f"Saved slice {slice_index} as {output_file}")

        # Add the slice index to the set of selected slices
        selected_slices.add(slice_index)
        saved_slices += 1

    if saved_slices < num_slices:
        print(f"Warning: Only {saved_slices}/{num_slices} slices were saved for {mha_file_path} due to uniform or duplicate slices.")

def extract_and_save_random_slices_from_directory(input_dir, output_dir, num_slices, max_files=None):
    """
    Extract random slices from a limited number of .mha files in a directory and its subdirectories.

    Parameters:
        input_dir (str): Directory containing .mha files.
        output_dir (str): Directory to save the extracted slices.
        num_slices (int): Number of random slices to extract from each file.output_file
        max_files (int): Maximum number of .mha files to process. If None, process all files.
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