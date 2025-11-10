import SimpleITK as sitk
import numpy as np
import os
from PIL import Image

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
input_dir = "/home/martina/repos/WP-BIO/model/nnunet/nnunet_raw/Dataset002_regiongrowing_qc_masked/imagesTs"
output_dir = "/home/martina/Dataset/rootpainter_inference_mask_png"
max_files = 145  # Process all training .mha files
process_directory(input_dir, output_dir, max_files=max_files)