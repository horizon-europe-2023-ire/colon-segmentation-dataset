"""
This file is used to download the CT COLONOGRAPHY dataset from the cancer archive.
Dicom images are saved in the tcia data folder in the raw subfolder.
We directly convert dicom images into .mha files as well and save them in the converted folder.

Scans with a 3rd dim < 350 and dim > 700 are directly removed.
Scans are further organized into subject subfolders.

USE:
You can either load the full dataset (takes more than 24 hours) by specifying files = None

You can download specific Instance-ids:
files = [
        "1.3.6.1.4.1.9328.50.81.87253217214149181807842257418975001776",
        "1.3.6.1.4.1.9328.50.4.693009",
        "1.3.6.1.4.1.9328.50.4.867016"
    ]

"""


import os
import requests
import zipfile
import tempfile
import SimpleITK as sitk
import shutil
import pandas as pd
import gzip
from pathlib import Path


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


base_dir = Path(__file__).resolve().parent
data_dir = os.path.join(base_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

meta_data_path = os.path.join(base_dir, 'meta_data_df.json')
meta_data_df = pd.read_json(meta_data_path, lines=True)

series_name_mapping = meta_data_df.set_index('InstanceUID')['name'].to_dict()
sub_id_mapping = meta_data_df.set_index('InstanceUID')['new_sub_id'].to_dict()

loaded_data_file = os.path.join(base_dir, 'loaded_data.txt')
if not os.path.exists(loaded_data_file):
    with open(loaded_data_file, 'w') as f:
        pass  # This creates an empty file
    print(f"File created: {loaded_data_file}")

loaded_data = read_txt_to_dict(loaded_data_file)
downloaded_series = set(loaded_data.keys())


def compress_file_gz(source_path, target_path):
    compressed_path = target_path + '.gz'
    try:
        with open(source_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb', compresslevel=1) as f_out:
                shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
        print(f"File compressed and saved as {compressed_path}")
        if os.path.exists(source_path):
            os.remove(source_path)
    except Exception as e:
        print(f"Error during compression: {e}")


def download_series(collection_name, files=None):
    # Ensure base and raw and converted directory exists
    raw_directory = os.path.join(data_dir, "raw")
    converted_directory = os.path.join(data_dir, "converted")

    os.makedirs(data_dir, exist_ok=True)
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
            if files:  # only load specific series defined in files
                if series_uid in files:
                    success = process_series(series_uid)
                    print(f"Downloaded series: {series_uid}")
                    # if series_uid not in downloaded_series:
                    with open(loaded_data_file, 'a') as f:
                        f.write(f"{series_uid} {success}\n")

            else:  # load all series that have not been downloaded
                if series_uid in downloaded_series:
                    print(f"Series {series_uid} already processed. Skipping...")
                    continue

                # Download and process the series
                success = process_series(series_uid)
                print(f"Processing series {series_uid}: {success}.")

                with open(loaded_data_file, 'a') as f:
                    f.write(f"{series_uid} {success}\n")

    else:
        print(f"Error fetching series for collection {collection_name}: {response.status_code}")


def process_series(series_uid):
    # Download and process the series
    download_endpoint = f"{BASE_URL}/getImage?SeriesInstanceUID={series_uid}"
    download_response = requests.get(download_endpoint, headers=headers, stream=True)

    converted_directory = os.path.join(data_dir, "converted")
    raw_directory = os.path.join(data_dir, "raw")

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
                if success:
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
            sub_id = sub_id_mapping.get(series_uid, 'NA')
            if sub_id == 'NA':
                raise ValueError(f"Subject ID for series: {series_uid} not available")

            filename = series_name_mapping.get(series_uid, "NA")
            if filename == 'NA':
                raise ValueError(f"Filename for series: {series_uid} not available")

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


def convert_manually_downloaded():
    folder = os.path.join("data/raw_manually")
    uid_series = os.listdir(folder)
    converted_directory = os.path.join(data_dir, "converted")
    raw_directory = os.path.join(data_dir, "raw")
    os.makedirs(converted_directory, exist_ok=True)
    os.makedirs(raw_directory, exist_ok=True)

    for uid_serie in uid_series:
        series_path = os.path.join(folder, uid_serie)
        success, subject_id, filename = convert_dicom_zip_to_mha(series_path, converted_directory, uid_serie)

        # save zipped dicom file
        if success:
            os.makedirs(os.path.join(raw_directory, subject_id), exist_ok=True)
            destination_path = os.path.join(raw_directory, f"{subject_id}/{filename}.zip")

            # if not zipfile.is_zipfile(series_path):
            #     shutil.make_archive(base_name=series_path, format='zip', root_dir=folder)
            #     shutil.rmtree(series_path)
            #     series_path = f"{series_path}.zip"
            #
            # shutil.move(series_path, destination_path)


def download(files):
    collection_name = "CT COLONOGRAPHY"
    download_series(collection_name, files=files)

