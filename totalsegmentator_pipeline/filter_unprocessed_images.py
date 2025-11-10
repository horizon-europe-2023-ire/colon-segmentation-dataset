import os

def get_completed_images(segmentation_dir):
    completed_images = set()
    for subject_folder in os.listdir(segmentation_dir):
        subject_path = os.path.join(segmentation_dir, subject_folder)
        if os.path.isdir(subject_path):  # Ensure it's a directory
            for scan_folder in os.listdir(subject_path):
                scan_path = os.path.join(subject_path, scan_folder)
                # Only add non-empty folders to the completed set
                if os.path.isdir(scan_path) and os.listdir(scan_path):
                    completed_images.add(scan_folder)
    return completed_images

def filter_unprocessed_images(image_paths_file, segmentation_dir, output_file):
    completed_images = get_completed_images(segmentation_dir)

    with open(image_paths_file, 'r') as f:
        all_image_paths = f.readlines()

    with open(output_file, 'w') as f:
        for path in all_image_paths:
            subject_id = path.split('/')[-2]  # Assumes subID is in the parent folder
            scan_id = os.path.basename(path).split('.mha.gz')[0]

            if f"{scan_id}" not in completed_images:
                f.write(path)

if __name__ == "__main__":
    image_paths_file = "/home/smp884/IRE/image_paths.txt"
    segmentation_dir = "/home/smp884/IRE/data/CT/segmentations_totalsegmentator"
    output_file = "/home/smp884/IRE/image_paths_filtered.txt"
    filter_unprocessed_images(image_paths_file, segmentation_dir, output_file)
