"""
===============================================================================
 Script: subtract_labelmaps.py
 Purpose:
     Extract the *fluid-only* region by subtracting one .mha labelmap from
     another — typically, subtracting the air-filled colon mask from the
     complete colon segmentation.

 Description:
     - Finds matching labelmap pairs in two folders based on their
       three-digit numeric identifiers (e.g., 001, 045, 127).
     - Subtracts labelmap2 (e.g., colon air-filled region) from labelmap1
       (e.g., full colon segmentation) to isolate the fluid component.
     - Preserves image geometry metadata (spacing, origin, direction).
     - Saves resulting binary masks (0/1) as .mha files to the output folder.

Notes:
⚠️ File Naming Requirement:
    Both folders must contain files with a common three-digit ID in
    their filenames — for example:

        Folder 1 (full colon):    colon_012_full.mha
        Folder 2 (air-filled):    colon_012_air.mha

    The script extracts that numeric ID ("012") and uses it to match
    corresponding files between folders. Files without matching IDs
    are skipped automatically.

 Usage:
     python subtract_labelmaps.py <folder_full_colon> <folder_colon_air> <output_folder>

 Example:
     python subtract_labelmaps.py \
         ../data/colon_full_masks \
         ../data/colon_air_masks \
         ../data/colon_fluid_masks
===============================================================================
"""

from pathlib import Path
import re
import numpy as np
import SimpleITK as sitk


def extract_number(filename: str) -> str | None:
    """Extract the first three-digit number from a filename."""
    match = re.search(r"\d{3}", filename)
    return match.group() if match else None


def subtract_labelmaps_mha(labelmap1_path: Path, labelmap2_path: Path, output_path: Path) -> None:
    """
    Subtract one labelmap from another and save the result as an .mha file.

    Parameters:
        labelmap1_path: Path to full colon segmentation (.mha)
        labelmap2_path: Path to colon air-filled segmentation (.mha)
        output_path:    Destination for resulting subtracted .mha file
    """
    labelmap1 = sitk.ReadImage(str(labelmap1_path))
    labelmap2 = sitk.ReadImage(str(labelmap2_path))

    array1 = sitk.GetArrayFromImage(labelmap1)
    array2 = sitk.GetArrayFromImage(labelmap2)

    if array1.shape != array2.shape:
        print(f"⚠️ Skipping due to shape mismatch: {labelmap1_path.name} vs {labelmap2_path.name}")
        return

    # Binary subtraction (ensures result is 0/1)
    result_array = ((array1 - array2) > 0).astype(np.uint8)
    result_image = sitk.GetImageFromArray(result_array)

    # Copy spatial metadata
    result_image.SetOrigin(labelmap1.GetOrigin())
    result_image.SetSpacing(labelmap1.GetSpacing())
    result_image.SetDirection(labelmap1.GetDirection())

    # Save
    sitk.WriteImage(result_image, str(output_path))
    print(f"✅ Saved: {output_path.name}")


def process_folders(source_folder1: Path, source_folder2: Path, output_folder: Path) -> None:
    """
    Match labelmap pairs by shared three-digit numbers and subtract one from another.

    Parameters:
        source_folder1: Path containing the full colon segmentations
        source_folder2: Path containing the colon air-filled segmentations
        output_folder:  Path to save the resulting fluid-only masks
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    files1 = {
        extract_number(f.name): f
        for f in source_folder1.glob("*.mha")
        if extract_number(f.name)
    }
    files2 = {
        extract_number(f.name): f
        for f in source_folder2.glob("*.mha")
        if extract_number(f.name)
    }

    common_ids = sorted(set(files1.keys()) & set(files2.keys()))
    if not common_ids:
        print("⚠️ No matching three-digit identifiers found between the two folders.")
        return

    print(f"Found {len(common_ids)} matching files. Starting subtraction...\n")

    for num in common_ids:
        file1 = files1[num]
        file2 = files2[num]
        output_path = output_folder / f"subtracted_{num}.mha"
        subtract_labelmaps_mha(file1, file2, output_path)

    print("\n✅ Subtraction complete.")


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python subtract_labelmaps.py <folder_full_colon> <folder_colon_air> <output_folder>")
        sys.exit(1)

    source_folder1 = Path(sys.argv[1]).resolve()
    source_folder2 = Path(sys.argv[2]).resolve()
    output_folder = Path(sys.argv[3]).resolve()

    print(f"Full colon folder: {source_folder1}")
    print(f"Air-filled folder: {source_folder2}")
    print(f"Output folder:     {output_folder}\n")

    process_folders(source_folder1, source_folder2, output_folder)
