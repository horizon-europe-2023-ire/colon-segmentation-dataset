"""
===============================================================================
 Script: resample_segmentations_to_original.py
 Purpose:
     Remap / resample segmentation .mha files so they match the original images'
     geometry (size, spacing, origin, direction). Useful when segmentations were
     produced on re-sampled volumes and must be aligned back to the originals.

 How it works:
     - For each segmentation file, extract the three-digit ID after "colon_"
       (e.g., "colon_012") and find the matching original .mha in the
       originals folder.
     - Resample the segmentation to the original image geometry using
       nearest-neighbor interpolation (to preserve labels).
     - Save to the output folder; filenames replace "v2" with "v3".

 Usage:
     python resample_segmentations_to_original.py <seg_folder> <orig_folder> <out_folder>

 Example:
     python resample_segmentations_to_original.py \
         ../data/seg_v2 \
         ../data/originals \
         ../data/seg_v3
===============================================================================
"""

from pathlib import Path
import re
import SimpleITK as sitk


def find_matching_original(segmentation_filename: str, original_folder: Path) -> Path | None:
    """
    Find the original image corresponding to the segmentation file by
    extracting the three-digit number after 'colon_'.
    """
    match = re.search(r"colon_(\d{3})", segmentation_filename)
    if not match:
        print(f"Warning: Could not extract number from {segmentation_filename}, skipping...")
        return None

    number = match.group(1)

    # Look for any .mha in originals containing that number
    for orig_path in original_folder.iterdir():
        if orig_path.suffix == ".mha" and number in orig_path.name:
            return orig_path

    print(f"Warning: No matching original image found for {segmentation_filename} (number {number}), skipping...")
    return None


def resample_segmentation_folder(segmentation_folder: Path, original_folder: Path, output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    for seg_path in segmentation_folder.iterdir():
        if seg_path.suffix != ".mha":
            continue

        original_image_path = find_matching_original(seg_path.name, original_folder)
        if original_image_path is None:
            continue  # Skip if no match found

        # Output filename: replace 'v2' with 'v3'
        output_filename = seg_path.name.replace("v2", "v3")
        output_path = output_folder / output_filename

        # Load images
        segmentation = sitk.ReadImage(str(seg_path))
        original_image = sitk.ReadImage(str(original_image_path))

        # Get metadata from original
        original_size = original_image.GetSize()
        original_spacing = original_image.GetSpacing()
        original_origin = original_image.GetOrigin()
        original_direction = original_image.GetDirection()

        # Define resampler (nearest-neighbor preserves label values)
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(original_spacing)
        resampler.SetSize(original_size)
        resampler.SetOutputOrigin(original_origin)
        resampler.SetOutputDirection(original_direction)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

        # Apply
        resampled_segmentation = resampler.Execute(segmentation)

        # Save
        sitk.WriteImage(resampled_segmentation, str(output_path))
        print(f"Processed: {seg_path.name} -> {output_filename}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python resample_segmentations_to_original.py <seg_folder> <orig_folder> <out_folder>")
        sys.exit(1)

    segmentation_folder = Path(sys.argv[1]).resolve()
    original_folder = Path(sys.argv[2]).resolve()
    output_folder = Path(sys.argv[3]).resolve()

    print(f"Segmentation folder: {segmentation_folder}")
    print(f"Originals folder:    {original_folder}")
    print(f"Output folder:       {output_folder}")

    resample_segmentation_folder(segmentation_folder, original_folder, output_folder)

    print("✅ Resampling completed.")
