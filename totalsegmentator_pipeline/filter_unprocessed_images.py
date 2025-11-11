"""
===============================================================================
 Script: filter_unprocessed_images.py
 Purpose:
     Identify which images from a list still need to be segmented and output
     a filtered list containing only those unprocessed image paths.

 Description:
     - The script reads a text file (e.g. image_paths.txt) containing full paths
       to all converted image volumes (.mha or .mha.gz).
     - It checks a given segmentation directory to see which scans already have
       segmentation results (non-empty folders).
     - It writes a new text file (e.g. image_paths_filtered.txt) listing only
       the images that still need to be processed.

 Example:
     Input:
         image_paths.txt:
             ../data/converted/sub001/scan1.mha.gz
             ../data/converted/sub002/scan2.mha.gz

         ../data/segmentations_totalsegmentator/
             ├── sub001/
             │    └── scan1/   ← already segmented
             └── sub002/       ← empty

     Output:
         image_paths_filtered.txt:
             ../data/converted/sub002/scan2.mha.gz

 Parameters:
     image_paths_file : Path
         Text file containing paths to all input image volumes.
     segmentation_dir  : Path
         Directory containing segmentation results (organized by subject/scan).
     output_file       : Path
         Destination file listing only unprocessed image paths.

===============================================================================
"""

from pathlib import Path

def get_completed_images(segmentation_dir: Path) -> set[str]:
    """
    Returns a set of scan IDs that already have non-empty segmentations.
    Ensures segmentation_dir exists.
    """
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    completed_images = set()

    for subject_path in segmentation_dir.iterdir():
        if subject_path.is_dir():
            for scan_path in subject_path.iterdir():
                # Add only non-empty directories
                if scan_path.is_dir() and any(scan_path.iterdir()):
                    completed_images.add(scan_path.name)

    return completed_images


def filter_unprocessed_images(image_paths_file: Path, segmentation_dir: Path, output_file: Path):
    completed_images = get_completed_images(segmentation_dir)

    all_image_paths = image_paths_file.read_text().splitlines()

    with output_file.open("w") as f_out:
        for path_str in all_image_paths:
            path = Path(path_str.strip())
            subject_id = path.parent.name
            scan_id = path.stem.replace(".mha", "").replace(".gz", "")

            if scan_id not in completed_images:
                f_out.write(f"{path_str}\n")

if __name__ == "__main__":
    # Relative paths
    image_paths_file = Path("image_paths.txt")
    segmentation_dir = Path("../data/segmentations_totalsegmentator")
    output_file = Path("image_paths_filtered.txt")

    filter_unprocessed_images(image_paths_file, segmentation_dir, output_file)


