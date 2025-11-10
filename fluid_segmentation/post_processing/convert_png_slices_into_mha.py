import os
import re
import numpy as np
from PIL import Image
import SimpleITK as sitk

def match_png_to_volumes(png_dir, reference_mha_dir, output_dir):
    """
    Converts PNG slices (output of rootpainter) back to .mha files, grouping by volumes based on matching names.

    Parameters:
        png_dir (str): Directory containing PNG slices named with volume identifiers.
        reference_mha_dir (str): Directory containing reference .mha files.
        output_dir (str): Directory to save the reconstructed .mha files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Group PNG files by volume identifier
    volume_slices = {}
    for png_file in os.listdir(png_dir):
        if png_file.endswith('.png'):
            # Extract the volume identifier from the PNG filename (e.g., colon_098)
            match = re.match(r"(.*_\d{4})", png_file)
            if match:
                volume_id = match.group(1)
                if volume_id not in volume_slices:
                    volume_slices[volume_id] = []
                volume_slices[volume_id].append(os.path.join(png_dir, png_file))

    # Process each volume
    for volume_id, png_files in volume_slices.items():
        # Sort PNG files numerically by slice index
        png_files.sort(key=lambda x: int(re.search(r'slice(\d+)', x).group(1)))

        # Find the corresponding reference .mha file
        reference_mha_file = os.path.join(reference_mha_dir, f"{volume_id}.mha")
        if not os.path.exists(reference_mha_file):
            print(f"Reference .mha file not found for volume: {volume_id}")
            continue

        # Load the reference .mha file
        reference_image = sitk.ReadImage(reference_mha_file)
        reference_spacing = reference_image.GetSpacing()
        reference_origin = reference_image.GetOrigin()
        reference_direction = reference_image.GetDirection()

        # Get original slice dimensions from the reference .mha file
        original_shape = sitk.GetArrayFromImage(reference_image).shape[1:]  # (Height, Width)

        # Load and process PNG slices
        slices = []
        for png_file in png_files:
            slice_image = Image.open(png_file).convert("L")  # Convert to grayscale

            # Resize the slice back to the original dimensions
            slice_image = slice_image.resize(original_shape[::-1], Image.Resampling.LANCZOS)  # Reverse shape for (width, height)

            # Convert to numpy array and create binary mask
            slice_array = np.array(slice_image)
            binary_mask = np.where(slice_array > 128, 255, 0).astype(np.uint8)
            #binary_mask = np.where(slice_array > 128, 1, 0).astype(np.uint8)  # Ensure binary values 0 and 1
            slices.append(binary_mask)


        # Stack slices into a 3D array
        image_array = np.stack(slices, axis=0)

        # Convert numpy array to SimpleITK image
        sitk_image = sitk.GetImageFromArray(image_array)

        # Set spacing, origin, and direction to match the reference image
        sitk_image.SetSpacing(reference_spacing)
        sitk_image.SetOrigin(reference_origin)
        sitk_image.SetDirection(reference_direction)

        # Save the reconstructed .mha file
        output_mha_path = os.path.join(output_dir, f"{volume_id}_fluidmask.mha")
        sitk.WriteImage(sitk_image, output_mha_path)
        print(f"Reconstructed .mha saved to: {output_mha_path}")

# usage
png_dir = "/home/martina/Dataset/root_painter_inference_mask_png_TsSet"  # Directory containing PNG slices
reference_mha_dir = "/home/martina/repos/WP-BIO/model/nnunet/nnunet_raw/Dataset002_regiongrowing_qc_masked/imagesTs/"  # Directory with reference .mha files
output_dir = "/home/martina/Dataset/rootpainter_inference_mask_mha/TestSet"  # Directory to save the reconstructed .mha files

match_png_to_volumes(png_dir, reference_mha_dir, output_dir)
