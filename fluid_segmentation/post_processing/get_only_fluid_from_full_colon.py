import SimpleITK as sitk
import numpy as np
import os
import re

#Get only the fluid and save into an .mha file from full colon segmentation

def extract_number(filename):
    """Extracts the first three-digit number from the filename."""
    match = re.search(r'\d{3}', filename)
    return match.group() if match else None

def subtract_labelmaps_mha(labelmap1_path, labelmap2_path, output_path):
    """Subtracts labelmap2 from labelmap1 and saves the result as an MHA file."""
    labelmap1 = sitk.ReadImage(labelmap1_path)
    labelmap2 = sitk.ReadImage(labelmap2_path)

    array1 = sitk.GetArrayFromImage(labelmap1)
    array2 = sitk.GetArrayFromImage(labelmap2)

    if array1.shape != array2.shape:
        print(f"Skipping {labelmap1_path} and {labelmap2_path} due to shape mismatch.")
        return
    
    # Ensure the subtraction results in a binary mask
    result_array = (array1 - array2) > 0  # Logical operation ensures binary values
    result_array = result_array.astype(np.uint8)  # Convert to uint8 (0 or 1)

    result_image = sitk.GetImageFromArray(result_array)

    # Preserve metadata
    result_image.SetOrigin(labelmap1.GetOrigin())
    result_image.SetSpacing(labelmap1.GetSpacing())
    result_image.SetDirection(labelmap1.GetDirection())

    sitk.WriteImage(result_image, output_path)
    print(f"Saved subtracted label map: {output_path}")

def process_folders(source_folder1, source_folder2, output_folder):
    """Matches files by three-digit numbers and subtracts them."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all .mha files from both folders
    files1 = {extract_number(f): os.path.join(source_folder1, f) for f in os.listdir(source_folder1) if f.endswith(".mha")}
    files2 = {extract_number(f): os.path.join(source_folder2, f) for f in os.listdir(source_folder2) if f.endswith(".mha")}

    # Find common numbers
    common_numbers = set(files1.keys()) & set(files2.keys())

    for num in common_numbers:
        file1 = files1[num]
        file2 = files2[num]
        output_path = os.path.join(output_folder, f"subtracted_{num}.mha")

        subtract_labelmaps_mha(file1, file2, output_path)

# Example usage
process_folders('/home/martina/Dataset/difficult_fluid', '/home/martina/repos/WP-BIO/model/nnunet/nnunet_raw/Dataset002_regiongrowing_qc_masked/labelsTr/', '/home/martina/Dataset/difficult_fluid/only_fluid/')
# Example usage
