"""
This file is used to download the CT COLONOGRAPHY dataset from the cancer archive.
Dicom images are saved in the tcia data folder in the raw subfolder.
We directly convert dicom images into .mha files as well and save them in the converted folder.

Scans with a 3rd dim < 350 and dim > 700 are directly removed.
Scans are further organized into subject subfolders.

USE:
You can either load the full dataset (takes more than 24 hours) by specifying files = None

You can put specific Instance-ids in files, to download only those.
"""


import os
import numpy as np
import requests
import zipfile
import tempfile
import SimpleITK as sitk
import shutil
import pydicom
import io
import json
import pandas as pd
import gzip

SAVE_LOCAL = True

# TCIA base URL for API requests
BASE_URL = "https://services.cancerimagingarchive.net/services/v4/TCIA/query"

# Headers for the request, including the API key for authentication
headers = {
    "api_key": "",
}


def read_txt_to_dict(file_path):
    data_dict = {}

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:  # Ensure there are two parts (seriesUID and subID)
                series_uid = parts[0]
                value = parts[1]
                data_dict[series_uid] = value  # Store in dict (overwrites duplicates)

    return data_dict


base_directory = os.path.join('..', "tcia-data")
meta_data_path = os.path.join(base_directory, 'meta-data/meta_data_df.json')
meta_data_df = pd.read_json(meta_data_path, lines=True)
sub_mapping_path = os.path.join(base_directory, 'meta-data/subject_id_mapping.txt')
sub_mapping = read_txt_to_dict(sub_mapping_path)
scan_mapping_path = os.path.join(base_directory, 'meta-data/scan_mapping.txt')
scan_mapping = read_txt_to_dict(scan_mapping_path)
loaded_data_file = os.path.join(base_directory, 'meta-data/loaded_data.txt')
loaded_data = read_txt_to_dict(loaded_data_file)
downloaded_series = set(loaded_data.keys())


def compress_file_gz(source_path, target_path):
    compressed_path = target_path + '.gz'
    try:
        with open(source_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"File compressed and saved as {compressed_path}")
    except Exception as e:
        print(f"Error during compression: {e}")


def download_series(collection_name, files=None):
    # Ensure base and raw and converted directory exists
    raw_directory = os.path.join(base_directory, "raw")
    converted_directory = os.path.join(base_directory, "converted")

    os.makedirs(base_directory, exist_ok=True)
    os.makedirs(raw_directory, exist_ok=True)
    os.makedirs(converted_directory, exist_ok=True)

    # API request to fetch series information
    endpoint = f"{BASE_URL}/getSeries"
    params = {"Collection": collection_name}
    response = requests.get(endpoint, headers=headers, params=params)

    if response.status_code == 200:
        series_list = response.json()
        for series in series_list:
            series_uid = series["SeriesInstanceUID"]
            if files:
                if series_uid in files:
                    success = process_series(series_uid, base_directory, meta_data_df, sub_mapping)
                    print(f"Downloaded series: {series_uid}")
                    # if series_uid not in downloaded_series:
                    with open(loaded_data_file, 'a') as f:
                        f.write(f"{series_uid} {success}\n")

            else:
                if series_uid in downloaded_series:
                    print(f"Series {series_uid} already processed. Skipping...")
                    continue

                # Download and process the series
                success = process_series(series_uid, base_directory, meta_data_df, sub_mapping)
                print(f"Processing series {series_uid}: {success}.")

                with open(loaded_data_file, 'a') as f:
                    f.write(f"{series_uid} {success}\n")

    else:
        print(f"Error fetching series for collection {collection_name}: {response.status_code}")


def process_series(series_uid, base_directory, meta_data_df, sub_mapping):
    # Download and process the series
    download_endpoint = f"{BASE_URL}/getImage?SeriesInstanceUID={series_uid}"
    download_response = requests.get(download_endpoint, headers=headers, stream=True)

    converted_directory = os.path.join(base_directory, "converted")
    raw_directory = os.path.join(base_directory, "raw")

    if download_response.status_code == 200:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, f"{series_uid}.zip")
                with open(file_path, "wb") as f:
                    for chunk in download_response.iter_content(chunk_size=128):
                        f.write(chunk)

                # Convert DICOM to MHA
                success, subject_id, filename = convert_dicom_zip_to_mha(file_path, converted_directory, series_uid)

                # save zipped dicom file
                if success and SAVE_LOCAL:
                    os.makedirs(os.path.join(raw_directory, subject_id), exist_ok=True)
                    destination_path = os.path.join(raw_directory, f"{subject_id}/{filename}.zip")
                    shutil.move(file_path, destination_path)
                return success

        except Exception as e:
            print(f"Error during download of series {series_uid}: {e}")
            return "ReadingFileError"
    else:
        print(f"Error downloading series {series_uid}: {download_response.status_code}")
        return "DownloadError"


def get_sub_id(dicom_names):
    dicom_data = pydicom.dcmread(dicom_names[0])
    sub_id = dicom_data.get((0x0010, 0x0020), "Not available")  # Patient's ID (Subject ID)
    if sub_id != "Not available":
        sub_id = sub_id.value
        return sub_mapping.get(sub_id, "NA")
    return "NA"


def get_position(dicom_names):
    dicom_data = pydicom.dcmread(dicom_names[0])
    patient_position = str(dicom_data.PatientPosition)  # Patient Position
    if patient_position == 'FFS' or patient_position == 'HFS':
        return "supine"
    elif patient_position == 'FFP' or patient_position == 'HFP':
        return "prone"
    return patient_position.lower()


def convert_dicom_zip_to_mha(file_path, output_directory, series_uid):
    try:
        # Check if file_path is a ZIP file and extract if necessary
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                temp_dir = tempfile.mkdtemp()
                zip_ref.extractall(temp_dir)
                file_to_read = temp_dir
        else:
            file_to_read = file_path

        # Read DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(file_to_read)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # Get image dimensions and DICOM metadata
        dimensions = image.GetSize()

        # Check if the dimensions meet the expected size range
        if all(350 < dim < 700 for dim in dimensions):
            # get subject id to get new subid
            sub_id = get_sub_id(dicom_names)

            # get scan number
            scan = scan_mapping.get(series_uid, 'NA')
            if scan == "NA":
                raise ValueError(f"Scan number for {series_uid} not available")

            # get position
            position = get_position(dicom_names)

            # create filename e.g. sub711_pos-supine_scan-1
            filename = f"{sub_id}_pos-{position}_scan-{scan}"

            if SAVE_LOCAL:
                # Create the output directory if it doesn't exist
                subject_dir = os.path.join(output_directory, sub_id)
                os.makedirs(subject_dir, exist_ok=True)
                # Create the .mha filename and save the file
                mha_filename = os.path.join(subject_dir, f"{filename}_conv-sitk.mha")
                sitk.WriteImage(image, mha_filename)
                # Compress the resulting .mha file using GZip
                compress_file_gz(mha_filename, mha_filename)
            return "Success", sub_id, filename

        else:
            return f"DimensionError {dimensions}", None, None
    except Exception as e:
        print(f"Error during conversion of {series_uid}: {e}")
        return f"ReadingFileError", None, None


if __name__ == "__main__":
    # Base directory to store all data
    collection_name = "CT COLONOGRAPHY"
    files = [
        "1.3.6.1.4.1.9328.50.81.87253217214149181807842257418975001776",
        "1.3.6.1.4.1.9328.50.4.693009",
        "1.3.6.1.4.1.9328.50.4.867016"
    ]

    download_series(collection_name, files=files)

