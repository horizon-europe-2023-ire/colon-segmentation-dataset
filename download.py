"""
This file downloads the CT COLONOGRAPHY dataset from TCIA (public, no API key).
- DICOM series are saved in data/raw/<subject_id>/<filename>.zip
- Each series is converted to .mha (gz) in data/converted/<subject_id>/
- Scans are skipped if the 3rd dim (z) < 350 or > 700.
- Scans are organized into subject subfolders.
- You can load the full dataset (files=None) or a list of SeriesInstanceUIDs.

Example:
files = [
    "1.3.6.1.4.1.9328.50.81.87253217214149181807842257418975001776",
    "1.3.6.1.4.1.9328.50.4.693009",
    "1.3.6.1.4.1.9328.50.4.867016",
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

# --- Public NBIA (TCIA) v1 base URL (no API key needed for public data) ---
BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"

# Request settings
REQUEST_TIMEOUT = (10, 600)   # (connect, read) -> allow large downloads
CHUNK_SIZE = 1024 * 1024      # 1 MB

def read_txt_to_dict(file_path):
    data_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                series_uid = parts[0]
                value = parts[1]
                data_dict[series_uid] = value
    return data_dict

base_dir = Path(__file__).resolve().parent
data_dir = os.path.join(base_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

meta_data_path = os.path.join(base_dir, 'metadata.jsonl')
meta_data_df = pd.read_json(meta_data_path, lines=True)

series_name_mapping = meta_data_df.set_index('InstanceUID')['name'].to_dict()
sub_id_mapping = meta_data_df.set_index('InstanceUID')['new_sub_id'].to_dict()

loaded_data_file = os.path.join(base_dir, 'loaded_data.txt')
if not os.path.exists(loaded_data_file):
    with open(loaded_data_file, 'w'):
        pass
    print(f"File created: {loaded_data_file}")

loaded_data = read_txt_to_dict(loaded_data_file)
downloaded_series = set(loaded_data.keys())

def compress_file_gz(source_path: str):
    """Compress a file to <path>.gz and remove the original."""
    compressed_path = source_path + '.gz'
    try:
        with open(source_path, 'rb') as f_in, gzip.open(compressed_path, 'wb', compresslevel=1) as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
        if os.path.exists(source_path):
            os.remove(source_path)
        print(f"Compressed -> {compressed_path}")
    except Exception as e:
        print(f"Error during compression: {e}")

def download_series(collection_name, files=None):
    # Ensure base/raw/converted directories exist
    raw_directory = os.path.join(data_dir, "raw")
    converted_directory = os.path.join(data_dir, "converted")
    os.makedirs(raw_directory, exist_ok=True)
    os.makedirs(converted_directory, exist_ok=True)

    if files:  # Only specific SeriesInstanceUIDs
        for series_uid in files:
            if series_uid in downloaded_series:
                print(f"Series {series_uid} already processed. Skipping...")
                continue
            success = process_series(series_uid)
            print(f"Processing series {series_uid}: {success}.")
            with open(loaded_data_file, 'a') as f:
                f.write(f"{series_uid} {success}\n")
        return

    # Full collection: fetch all series metadata (may be large; consider batching by patient if needed)
    endpoint = f"{BASE_URL}/getSeries"
    params = {"Collection": collection_name, "format": "json"}
    try:
        response = requests.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching series for collection '{collection_name}': {e}")
        return

    series_list = response.json()
    for series in series_list:
        series_uid = series.get("SeriesInstanceUID")
        if not series_uid:
            continue
        if series_uid in downloaded_series:
            print(f"Series {series_uid} already processed. Skipping...")
            continue

        success = process_series(series_uid)
        print(f"Processing series {series_uid}: {success}.")
        with open(loaded_data_file, 'a') as f:
            f.write(f"{series_uid} {success}\n")

def process_series(series_uid):
    """Download a series ZIP via v1 API and convert to .mha.gz if z in [350,700]."""
    download_endpoint = f"{BASE_URL}/getImage"
    params = {"SeriesInstanceUID": series_uid}

    converted_directory = os.path.join(data_dir, "converted")
    raw_directory = os.path.join(data_dir, "raw")

    try:
        with requests.get(download_endpoint, params=params, stream=True, timeout=REQUEST_TIMEOUT) as r:
            if r.status_code != 200:
                print(f"Error downloading series {series_uid}: HTTP {r.status_code}")
                return "DownloadError"

            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, f"{series_uid}.zip")
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

                # Convert DICOM to MHA
                success, subject_id, filename = convert_dicom_zip_to_mha(zip_path, converted_directory, series_uid)

                # Save zipped DICOM file in raw/<subject>/<filename>.zip on success
                if success == "Success":
                    os.makedirs(os.path.join(raw_directory, subject_id), exist_ok=True)
                    destination_path = os.path.join(raw_directory, f"{subject_id}/{filename}.zip")
                    shutil.move(zip_path, destination_path)
                return success

    except Exception as e:
        print(f"Error during download of series {series_uid}: {e}")
        return "DownloadException"

def convert_dicom_zip_to_mha(file_path, output_directory, series_uid):
    """Extract a series ZIP, read the DICOM stack, filter by slice count, write .mha.gz."""
    temp_dir = None
    try:
        # Must be a ZIP
        if not zipfile.is_zipfile(file_path):
            return "ReadingFileError", None, None

        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Read DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(temp_dir)
        if not dicom_names:
            return "NoDICOMFound", None, None

        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # Dimensions: (x, y, z). Enforce z (3rd dim) rule only.
        size = image.GetSize()
        z = size[2]
        if z < 350 or z > 700:
            return f"DimensionError z={z}", None, None

        # Resolve subject/filename from your metadata mappings
        sub_id = sub_id_mapping.get(series_uid, 'NA')
        if sub_id == 'NA':
            raise ValueError(f"Subject ID for series {series_uid} not available in metadata.jsonl")
        filename = series_name_mapping.get(series_uid, "NA")
        if filename == 'NA':
            raise ValueError(f"Filename for series {series_uid} not available in metadata.jsonl")

        # Write .mha then gzip it
        subject_dir = os.path.join(output_directory, sub_id)
        os.makedirs(subject_dir, exist_ok=True)
        mha_path = os.path.join(subject_dir, f"{filename}_conv-sitk.mha")
        sitk.WriteImage(image, mha_path)
        compress_file_gz(mha_path)

        return "Success", sub_id, filename

    except Exception as e:
        print(f"Error during conversion of {series_uid}: {e}")
        return "ReadingFileError", None, None
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

def convert_manually_downloaded():
    folder = os.path.join("data/raw_manually")
    if not os.path.isdir(folder):
        print(f"No folder found at {folder}")
        return

    converted_directory = os.path.join(data_dir, "converted")
    raw_directory = os.path.join(data_dir, "raw")
    os.makedirs(converted_directory, exist_ok=True)
    os.makedirs(raw_directory, exist_ok=True)

    for entry in os.listdir(folder):
        series_uid = os.path.splitext(entry)[0]
        series_path = os.path.join(folder, entry)
        success, subject_id, filename = convert_dicom_zip_to_mha(series_path, converted_directory, series_uid)
        if success == "Success":
            os.makedirs(os.path.join(raw_directory, subject_id), exist_ok=True)
            destination_path = os.path.join(raw_directory, f"{subject_id}/{filename}.zip")
            # If you want to move/copy the original ZIP into raw as well, uncomment:
            # shutil.move(series_path, destination_path)

def download(files):
    collection_name = "CT COLONOGRAPHY"
    download_series(collection_name, files=files)

if __name__ == "__main__":
    files = [
        "1.3.6.1.4.1.9328.50.4.850207",
        "1.3.6.1.4.1.9328.50.4.849142"
    ]
    download(files)
