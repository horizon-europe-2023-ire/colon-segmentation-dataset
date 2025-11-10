import os
import SimpleITK as sitk
import gzip
from pathlib import Path
from totalsegmentator.python_api import totalsegmentator
import shutil

def decompress_mha_gz(input_path, output_path):
    """Decompress a .mha.gz file to .mha without loading it into memory with SimpleITK."""
    with gzip.open(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def get_temp_filename(input_file_path, extension=".nii.gz"):
    """Generate a unique temporary filename based on the input path."""
    # Remove .mha, .mha.gz, or .mha.zip extensions
    base_name = Path(input_file_path).name
    temp_name = base_name.replace(".mha.gz", "").replace(".mha.zip", "").replace(".mha", "")

    # Append the desired extension (e.g., .nii.gz)
    return f"{temp_name}_temp{extension}"

def mha_to_nii(input_mha_file_path, temp_nii_file_path):
    """Convert MHA file to NIfTI format using SimpleITK and save temporarily."""
    image = sitk.ReadImage(input_mha_file_path)
    sitk.WriteImage(image, temp_nii_file_path)

def nii_to_mha_gz(input_nii_file_path, output_mha_gz_file_path):
    """Convert NIfTI file back to MHA, then compress to MHA.gz."""
    # Convert NIfTI to MHA
    mha_path = output_mha_gz_file_path.replace(".mha.gz", ".mha")
    image = sitk.ReadImage(input_nii_file_path)
    sitk.WriteImage(image, mha_path)

    # Compress the .mha file to .mha.gz
    with open(mha_path, 'rb') as f_in:
        with gzip.open(output_mha_gz_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Remove the uncompressed .mha file after compression
    os.remove(mha_path)

def ensure_directory_exists(path):
    """Ensure the directory exists, and create it if it doesn't."""
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def rename_totalseg_output(output_dir, output_base_path):
    """Rename TotalSegmentator output files according to specified convention."""
    folder_name = Path(output_base_path).stem
    for nii_file in os.listdir(output_dir):
        if nii_file.endswith(".nii.gz"):
            organ_name = Path(nii_file).stem.replace(".nii", "")
            new_name = f"{folder_name}_totalseg-{organ_name}.nii.gz"
            new_path = os.path.join(output_dir, new_name)
            os.rename(os.path.join(output_dir, nii_file), new_path)

def rename_and_convert_totalseg_output(output_dir, output_base_path):
    """Rename TotalSegmentator output files according to specified convention and compress to .mha.gz."""
    folder_name = Path(output_base_path).stem
    for nii_file in os.listdir(output_dir):
        if nii_file.endswith(".nii.gz"):
            organ_name = Path(nii_file).stem.replace(".nii", "")
            new_name = f"{folder_name}_totalseg-{organ_name}.mha.gz"
            new_path = os.path.join(output_base_path, new_name)

            # Convert each NIfTI organ file to .mha.gz
            nii_file_path = os.path.join(output_dir, nii_file)
            nii_to_mha_gz(nii_file_path, new_path)

def run_totalsegmentator(input_mha_file_path, output_base_path):
    """Convert MHA to NIfTI, run TotalSegmentator, and convert back to MHA."""

    # Step 1: Convert .mha.gz to .mha by decompressing it
    decompressed_mha_path = input_mha_file_path.replace(".mha.gz", ".mha")
    decompress_mha_gz(input_mha_file_path, decompressed_mha_path)

    # Step 2: Convert .mha to temporary .nii.gz
    temp_input_nii = get_temp_filename(decompressed_mha_path, ".nii.gz")
    mha_to_nii(decompressed_mha_path, temp_input_nii)

    # Step 3: Define temporary output directory for TotalSegmentator results
    temp_output_dir = get_temp_filename(output_base_path, "")  # Temporary directory for output

    try:
        # Step 4: Run TotalSegmentator on the temporary NIfTI file
        totalsegmentator(temp_input_nii, temp_output_dir)

        # Step 5: Handle TotalSegmentator output (directory): Rename and convert .nii.gz to mha.gz
        #rename_totalseg_output(temp_output_dir, output_base_path)
        rename_and_convert_totalseg_output(temp_output_dir, output_base_path)

    finally:
        # Clean up temporary files and directories
        if os.path.exists(decompressed_mha_path):
            os.remove(decompressed_mha_path)
        #if os.path.exists(temp_input_nii):
        #    os.remove(temp_input_nii)
        elif os.path.exists(temp_output_dir) and os.path.isdir(temp_output_dir):
            shutil.rmtree(temp_output_dir)  # Remove directory and its contents
        #elif os.path.exists(temp_output_nii):
        #    os.remove(temp_output_nii)  # Remove file if it's not a directory

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_mha_file_path> <output_mha_file_path>")
        sys.exit(1)

    input_mha_file_path = sys.argv[1]
    output_mha_file_path = sys.argv[2]

    run_totalsegmentator(input_mha_file_path, output_mha_file_path)
