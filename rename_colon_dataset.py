"""
File Renaming Utility — Naming Convention Synchronization
=============================================================

This script helps synchronize and rename files across different naming conventions 
used in the Colon Dataset project. During dataset creation and segmentation, 
different identifiers are used for CT scans, segmentations, and meshes. 

This script automates renaming between these systems using a mapping file 
(`naming.jsonl`) that records each scan’s identifiers across conventions.

-------------------------------------------------------------
 Supported Naming Conventions
-------------------------------------------------------------
1. InstanceUID
    - The unique DICOM identifier from the TCIA CT Colonography dataset.
    - Each scan (DICOM series) has a unique InstanceUID.

2. name
    - Custom human-readable convention used throughout the HQColon pipeline.
    - Example: "sub030_pos-prone_scan-2"
      → subject: sub030, position: prone, scan number: 2

3. colon_name
    - Short numerical naming (e.g., "colon_432") used for nnU-Net training.
    - If colon_name is null, that scan has no successful segmentation.

-------------------------------------------------------------
Required Input
-------------------------------------------------------------
- A `naming.jsonl` file containing all naming mappings. Example:

    {"InstanceUID": "1.3.6.1.4.1.9328.50.4.850207", 
     "name": "sub030_pos-prone_scan-2", 
     "colon_name": "colon_432"}

    {"InstanceUID": "1.3.6.1.4.1.9328.50.4.849142", 
     "name": "sub031_pos-prone_scan-1", 
     "colon_name": "colon_433"}

-------------------------------------------------------------
Main Functions
-------------------------------------------------------------
- get_name(wanted_naming_conv, given_naming_conv, name)
    Converts a file name from one naming convention to another.

- rename_file(folder_path, new_name, old_name)
    Renames a single file safely, checking for conflicts.

- rename_files_in_folder(wanted_naming_conv, given_naming_conv, folder)
    Batch-renames all .mha/.mha.gz files in a folder.

-------------------------------------------------------------
Example Usage
-------------------------------------------------------------
# Convert between conventions
name = 'colon_419.mha'
new_name = get_name('InstanceUID', 'colon_name', name)
print(new_name)

# Rename a single file
rename_file('data', 'sub030_pos-prone_scan-2', 'colon_419.mha')

# Rename all files in a folder
rename_files_in_folder('colon_name', 'name', 'data')

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
        print(f"Error: could not find {name} in dataset.")
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
    # Example usage: Convert a name and print the result
    name = 'colon_419.mha'
    new_name = get_name('InstanceUID', 'colon_name', name)
    print(new_name)

    # # Example: Rename files
    # folder = 'data'
    # rename_file(folder, new_name, name)
    #
    # # Example: Batch rename all files in a folder
    # rename_files_in_folder('colon_name', 'name', folder)




