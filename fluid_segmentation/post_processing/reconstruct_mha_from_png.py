"""
===============================================================================
 Script: reconstruct_mha_from_png.py
 Purpose:
     Reconstruct full 3D .mha medical image volumes from 2D PNG segmentation
     slices (e.g., masks produced by RootPainter).

 Description:
     - Takes PNG slices created from RootPainter inference or annotation.
     - Groups PNGs by their volume identifier inferred from file names.
     - Sorts them by slice index to preserve spatial order.
     - Finds the corresponding reference .mha volume to copy its metadata
       (spacing, origin, direction).
     - Stacks the PNGs back into a 3D array and saves as a new .mha file
       with the same geometry as the reference volume.

 Usage:
     python reconstruct_mha_from_png.py <png_dir> <reference_mha_dir> <output_dir>

 Example:
     python reconstruct_mha_from_png.py \
         ../data/rootpainter_inference_pngs \
         ../data/converted \
         ../data/rootpainter_reconstructed_mha

 Notes:
     - PNG filenames must include a volume identifier and a slice index,
       for example: "colon_0001_slice012.png".
     - The corresponding reference .mha file must have the same base name,
       e.g. "colon_0001.mha".
     - Output files are saved as "<volume_id>_fluidmask.mha".
===============================================================================
"""

import re
import numpy as np
from PIL import Image
import SimpleITK as sitk
from pathlib import Path


def match_png_to_volumes(png_dir: Path, reference_mha_dir: Path, output_dir: Path):
    """
    Converts PNG slices (output of RootPainter) back to .mha volumes,
    grouping by volume identifiers and preserving reference image metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group PNG files by volume identifier
    volume_slices: dict[str, list[Path]] = {}
    for png_file in png_dir.glob("*.png"):
        match = re.match(r"(.*_\d{4})", png_file.name)
        if match:
            volume_id = match.group(1)
            volume_slices.setdefault(volume_id, []).append(png_file)

    # Process each volume group
    for volume_id, png_files in volume_slices.items():
        # Sort PNG files by slice index (e.g., slice012)
        png_files.sort(key=lambda p: int(re.search(r"slice(\d+)", p.name).group(1)))

        reference_mha_file = reference_mha_dir / f"{volume_id}.mha"
        if not reference_mha_file.exists():
            print(f"⚠️  Reference .mha not found for volume: {volume_id}")
            continue

        # Load reference image metadata
        reference_image = sitk.ReadImage(str(reference_mha_file))
        ref_spacing = reference_image.GetSpacing()
        ref_origin = reference_image.GetOrigin()
        ref_direction = reference_image.GetDirection()
        ref_shape = sitk.GetArrayFromImage(reference_image).shape[1:]  # (H, W)

        # Read and resize PNG slices
        slices = []
        for png_path in png_files:
            slice_img = Image.open(png_path).convert("L")  # grayscale
            slice_img = slice_img.resize(ref_shape[::-1], Image.Resampling.LANCZOS)
            slice_arr = np.array(slice_img)
            binary_mask = np.where(slice_arr > 128, 255, 0).astype(np.uint8)
            slices.append(binary_mask)

        # Stack into 3D volume
        image_array = np.stack(slices, axis=0)
        sitk_image = sitk.GetImageFromArray(image_array)

        # Copy geometry
        sitk_image.SetSpacing(ref_spacing)
        sitk_image.SetOrigin(ref_origin)
        sitk_image.SetDirection(ref_direction)

        # Save reconstructed .mha
        output_mha_path = output_dir / f"{volume_id}_fluidmask.mha"
        sitk.WriteImage(sitk_image, str(output_mha_path))
        print(f"✅ Reconstructed: {output_mha_path}")


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python reconstruct_mha_from_png.py <png_dir> <reference_mha_dir> <output_dir>")
        sys.exit(1)

    png_dir = Path(sys.argv[1]).resolve()
    reference_mha_dir = Path(sys.argv[2]).resolve()
    output_dir = Path(sys.argv[3]).resolve()

    print(f"PNG directory:         {png_dir}")
    print(f"Reference MHA dir:     {reference_mha_dir}")
    print(f"Output directory:      {output_dir}")
    print("Starting reconstruction...\n")

    match_png_to_volumes(png_dir, reference_mha_dir, output_dir)

    print("\n✅ All matching volumes processed successfully.")