from helper import *
import pandas as pd
import os


erda_folder = '/home/amin/ucph-erda-home/IRE-DATA/CT'
base_folder = '/home/amin/PycharmProjects/WP-BIO'


def list_gz_files(folder_path):
    gz_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.gz'):
                base_name = os.path.basename(file)
                base_name = '_'.join(base_name.split('_')[:4])
                gz_files.append(base_name)
    return gz_files


if __name__ == "__main__":
    converted_folder = os.path.join(erda_folder, "converted")
    segmentations_colon_folder = os.path.join(base_folder, "tcia-data", "segmentations", "segmentations-regionalgrowing")
    plots_folder = os.path.join(base_folder, "tcia-data", "plots")
    os.makedirs(segmentations_colon_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    # Check if the 'converted' folder exists
    if not os.path.isdir(converted_folder):
        print(f"The directory {converted_folder} does not exist. No data to segment.")
        exit()

    # List all files in the 'converted' folder
    # file names: sub002_pos-prone_scan-1_conv-sitk_thr-n800_nei-1.mha.gz
    existing_seg_files = list_gz_files(segmentations_colon_folder)

    threshold = -800
    neighbours = 1
    # meta_data_df = pd.read_json(os.path.join(erda_folder, "metadata", "meta_data_df.json"), lines=True)
    meta_data_df = pd.read_json(os.path.join(base_folder, "tcia-data", "meta-data", "meta_data_df.json"), lines=True)

    ids = [
        "1.3.6.1.4.1.9328.50.4.434283"
    ]

    if ids:
        df = meta_data_df[meta_data_df['InstanceUID'].isin(ids)]
        filenames = df.name.tolist()
        # sub091_pos-supine_scan-1_conv-sitk.mha.gz"
        filenames = [f"{name}_conv-sitk" for name in filenames]

    else:
        # segment all
        filenames = list_gz_files(converted_folder)

    for file in filenames:

        filename = file.split('conv')[0][:-1]
        if filename in existing_seg_files:
            # indicated that that file was already processed, skip to next file
            print(f"Skipping {filename}")
            continue

        subject = file.split('_')[0]

        # source_path = f"{converted_folder}/{subject}/{file}"
        source_path = os.path.join(erda_folder, 'converted', subject, f"{file}.mha.gz")
        target_directory = os.path.join(segmentations_colon_folder, subject)
        os.makedirs(target_directory, exist_ok=True)

        existing_segmentations = os.listdir(target_directory)
        is_present = any(file in s for s in existing_segmentations)
        if is_present:
            print(f"Already segmented: {file}")
            continue

        print(f"Start to segment: {file}")
        segmented_filepath = create_segmentation(source_path, target_directory, file, threshold=threshold,
                                                 neighbours=neighbours,
                                                 coloured=True, only_thresholding=False)

        if segmented_filepath is None:
            continue

        compress_file(segmented_filepath, segmented_filepath)
        delete_file(segmented_filepath)
