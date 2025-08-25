"""
This script helps to rename files and swap between different naming conventions. It needs the file naming.jsonl where all names are saved in the different conventions.
Different naming convention have been used for segmentation, meshes and CT images. In general we differ between the following three name types:
- InstanceUID: This id stems from the TCIA CT-Colonography dataset. Each scan has a unique instance uid
- name: This naming stems from our own naming convention to ensure transparency and reproducibility. While we extended those filenames for each step of our pipeline (e.g. region growing to segment the air we add: thr-n800 or nei-1 to indicate what parameters we used to optain this segmentation. But the beginning is always the same:
    "sub030_pos-prone_scan-2" indicates our unique subject sub030, the position of the patient during the scan pos-prone and the scan number scan-2 as we can have multiple scans per patient.
- colon_name: This naming stems from training an nnU-Net. E.g. colon_432 assigns a unique 3-digit number to each file. If colon_name is null, then there exists no successfull segmentation for that scan.
This script helps to rename files from one to another naming convention.
"""

import pandas as pd
import os


# Path to your .jsonl file
file_path = "naming.jsonl"

# Read into DataFrame
df = pd.read_json(file_path, lines=True)


def get_name(wanted_naming_conv, given_naming_conv, name):
    name = name.replace('.mha', '')
    name = name.replace('.gz', '')

    if given_naming_conv == 'name':
        idx = name.find("scan-")
        idx += len("scan-") + 1
        name = name[:idx]

    row = df[df[given_naming_conv].isin([name])]
    if len(row) == 1:
        desired_name = row.loc[row.index[0], wanted_naming_conv]
        return desired_name
    else:
        print(f"Error: could not find {name} in  dataset.")
        return None


def rename_file(folder_path, new_name, old_name):
    if '.mha' in old_name:
        idx = old_name.find(".mha")
        extension = old_name[idx:]
    else:
        print(f"Error: {old_name} has to be an .mha file")
        return False

    new_name = f"{new_name}{extension}"

    new_path = os.path.join(folder_path, new_name)
    old_path = os.path.join(folder_path, old_name)

    if not os.path.isdir(os.path.join(folder_path)):
        print(f"Error: Folder {old_path} does not exist.")
        return False

    if not os.path.isfile(old_path):
        print(f"Error: File {old_path} does not exist.")
        return False

    if os.path.isfile(new_path):
        print(f"Error: File {new_path} already exists. Make sure you remove it first.")
        return False

    os.rename(old_path, new_path)
    print(f"Renamed: {old_path} to {new_path}")
    return True


def rename_files_in_folder(wanted_naming_conv, given_naming_conv, folder):
    if not os.path.isdir(os.path.join(folder)):
        print(f"Error: Folder {folder} does not exist.")
        return False

    files = os.listdir(os.path.join(folder))
    files = [f for f in files if f.endswith((".mha", ".mha.gz"))]
    for file in files:
        new_name = get_name(wanted_naming_conv, given_naming_conv, file)
        if new_name:
            rename_file(folder, new_name, file)


if __name__ == "__main__":
    # how to get the InstanceUID for a given name
    name = 'colon_419.mha'
    new_name = get_name('InstanceUID', 'colon_name', name)
    print(new_name)

    # # how to rename files
    # folder = 'data'
    # rename_file(folder, new_name, name)
    #
    # # rename all files in a folder -> files must end with .mha or .mha.gz
    # rename_files_in_folder('colon_name', 'name', folder)



