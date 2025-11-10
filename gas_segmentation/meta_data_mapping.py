import io
import os
import tempfile
import zipfile
import SimpleITK as sitk
import pandas as pd
import pydicom
import requests
import json
from collections import Counter

api_url = "https://services.cancerimagingarchive.net/services/v4/TCIA/query/"

subjects = []


def get_meta_data_from_file():
    with open("tcia-data/meta-data/info_data.json", 'r') as json_file:
        data = json.load(json_file)
    return data


def create_meta_data_mapping():
    # Define the API URL and Series Instance UID

    # Define the parameters
    endpoint_series = f"{api_url}getSeries"
    endpoint_patients = f"{api_url}getPatient"
    params = {
        "Collection": "CT COLONOGRAPHY"
    }

    response_series = requests.get(endpoint_series, params=params)
    response_patients = requests.get(endpoint_patients, params=params)

    if response_series.status_code == 200 and response_patients.status_code == 200:
        series_list = response_series.json()
        subject_list = response_patients.json()

        patient_data = {}
        no_gender = 0
        # Loop through the subjects and extract relevant information
        for subject in subject_list:
            # Extract patient-level information
            patient_id = subject.get("PatientID", "Not available")
            subjects.append(patient_id)
            patient_sex = subject.get("PatientSex", "Not available")

            if patient_sex == "Not available":
                no_gender += 1

            # Append extracted information into the list
            patient_data[patient_id] = patient_sex

        print(f"There were {no_gender} subjects without gender.")

        series_data = []

        metadata = get_meta_data_from_file()
        mismatched_positions = {}

        for series in series_list:
            series_uid = series.get("SeriesInstanceUID", "Not available")
            subject_id = series.get("PatientID", "Not available")
            if series_uid in metadata.keys():
                patient_position = metadata[series_uid]['patient_position']
                if "p" in patient_position.lower():
                    patient_position = "Prone"
                elif "s" in patient_position.lower():
                    patient_position = "Supine"
            else:
                patient_position = "Not available"
            study_description = series.get("SeriesDescription", "Not available")
            if study_description == "Not available":
                study_description = series.get("ProtocolName", "Not available")
            position = "Not available"
            if study_description != "Not available":
                if "prone" in study_description.lower():
                    position = "Prone"
                elif "supine" in study_description.lower():
                    position = "Supine"
            if position == "Not available":
                # In cases the description is not available we take the patient position
                if patient_position == "Not available":
                    print(f"Error: No position found for Series: {series_uid}, Sub: {subject_id}")
                else:
                    position = patient_position
            else:
                # In cases where description suggests one but patient position the other we take the description,
                # unless:
                if patient_position == "HFDR" or patient_position == "FFDL" or patient_position == "FFDR":
                    # for cases where the patient position suggests the patient is lying on the side we take that
                    position = patient_position
            if position != "Not available" and patient_position != "Not available" and position != patient_position:
                mismatched_positions[series_uid] = {"subject": subject_id, "description": position, "patient position": patient_position}

            series_date = series.get("SeriesDate", "Not available")

            if series_uid == "Not available" or subject_id == "Not available":
                print(f"Assertion Error: series Uid {series_uid} or Subject ID {subject_id} not defined")

            if subject_id in patient_data.keys():
                series_data.append({"InstanceUID": series_uid, "SubjectID": subject_id, "Date": series_date,
                                    "Position": position, "Sex": patient_data[subject_id]})
            else:
                print(f"No patient data found for subject {subject_id}")

        with open('tcia-data/mapping/map_data.json', 'w') as json_file:
            json.dump(series_data, json_file)
        with open('tcia-data/mapping/mismatched_positions.json', 'w') as json_file:
            json.dump(mismatched_positions, json_file)
    else:
        print("Request Unsuccessful.")


def get_mha_files_on_erda():
    filename = 'tcia-data/mapping/converted_files_list.txt'
    erda_files = {}
    with open(filename, 'r') as file:
        for line in file:
            filename = line.split("/")[-1]
            name = filename.split("_")[0]
            erda_files[name] = filename.strip()
    return erda_files


def get_series_with_dim_error():
    filename = 'tcia-data/meta-data/loaded_data.txt'
    series_with_dim_errors = []
    with open(filename, 'r') as file:
        for line in file:
            splits = line.split(" ")
            series = splits[0]
            if len(splits) == 3:
                if splits[1] == "Dimension":
                    series_with_dim_errors.append(series)
    return series_with_dim_errors


def create_sub_id():
    df = pd.read_json('tcia-data/mapping/map_data.json')

    df.loc[df['Sex'] == "Not Available"]['Sex'] = "NA"

    numb = len(df)
    erda_files = get_mha_files_on_erda

    df = df[df['InstanceUID'].isin(erda_files.keys())]
    print(f"We dropped {numb - len(df)} series as they are not on ERDA. There are {len(df)} left.")

    unique_subject_ids = list(df['SubjectID'].unique())

    subject_name_mapping = {subject_id: f'sub{str(index).zfill(3)}' for index, subject_id in
                            enumerate(unique_subject_ids, start=1)}

    df['new_subject_id'] = df['SubjectID'].map(subject_name_mapping)

    df['version'] = 1

    for subject_id in unique_subject_ids:
        # Get rows for the current SubjectID
        subject_rows = df[df['SubjectID'] == subject_id]

        for pos in list(df['Position'].unique()):
            pos_rows = subject_rows[subject_rows['Position'] == pos]

            if len(pos_rows) > 1:
                for index, (idx, row) in enumerate(pos_rows.iterrows()):
                    instance_id = row['InstanceUID']
                    df.loc[df['InstanceUID'] == instance_id, 'version'] = index + 1

    df['old_mha_path'] = df.apply(lambda row: f"converted/{erda_files.get(str(row['InstanceUID']), 'Path Not Found')}", axis=1)
    df['name'] = df.apply(lambda row: f"{row['new_subject_id']}_{row['Sex']}_{row['Position']}_{row['version']}", axis=1)
    df['new_mha_path'] = df.apply(lambda row: f"converted/{row['new_subject_id']}/{row['name']}_sitk.mha.zip", axis=1)
    df['old_dicom_path'] = df.apply(lambda row: f"raw/{row['InstanceUID']}.zip", axis=1)
    df['new_dicom_path'] = df.apply(lambda row: f"raw/{row['new_subject_id']}/{row['name']}.zip", axis=1)

    assert len(df) == len(df['name'].unique())

    contains_not_available = df[df['Position'].str.contains("Not available", na=False)]
    assert len(contains_not_available['InstanceUID']) == 0

    existing_ids = set(df['InstanceUID'])
    missing_ids = [id for id in erda_files.keys() if id not in existing_ids]
    assert len(missing_ids) == 0

    missing_subject = [sub for sub in subjects if sub not in unique_subject_ids]
    assert len(missing_subject) == 0

    df.to_json('tcia-data/mapping/name_mapping_df.json', orient='records', lines=True)


def get_dicom_info(zip_filename):
    try:
        # Extract the contents of the ZIP file to a temporary directory
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            temp_dir = tempfile.mkdtemp()
            zip_ref.extractall(temp_dir)

        # Read DICOM series from the temporary directory
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(temp_dir)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # Get the dimensions of the image
        dimensions = image.GetSize()

        # Read the first DICOM file for metadata
        dicom_data = pydicom.dcmread(dicom_names[0])

        # Extract relevant DICOM metadata
        dicom_header_info = {
            "dimensions": list(dimensions),
            "iop_list": list(dicom_data.ImageOrientationPatient) if 'ImageOrientationPatient' in dicom_data else None,
            "patient_position": str(dicom_data.PatientPosition) if 'PatientPosition' in dicom_data else None,
            "pixel_spacing": list(dicom_data.PixelSpacing) if 'PixelSpacing' in dicom_data else None,
            "slice_thickness": float(dicom_data.SliceThickness) if 'SliceThickness' in dicom_data else None,
            "image_position_patient": list(dicom_data.ImagePositionPatient) if 'ImagePositionPatient' in dicom_data else None
        }

        return dicom_header_info

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None


def find_dicom_path(name):
    dir_path = r'Z:/IRE-DATA/CT/raw'
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if name in file:
                return file
    return None


def find_missing_information(row):
    # find path on erda
    name = row['name']
    file = find_dicom_path(name)
    if file:
        # load dicom and get dicom header info
        data = get_dicom_info(file)
        # update row
        row['dimensions'] = data.get('dimensions', None)
        row['iop_list'] = data.get('iop_list', None)
        row['patient_position'] = data.get('patient_position', None)
        row['pixel_spacing'] = data.get('pixel_spacing', None)
        row['slice_thickness'] = data.get('slice_thickness', None)
        row['image_position_patient'] = data.get('image_position_patient', None)
    else:
        print(f"No DICOM image found for {name}")
    return row


def find_additional_information():
    additional_data = get_meta_data_from_file()
    df = pd.read_json('tcia-data/mapping/name_mapping_df.json', lines=True)
    erda_files = get_mha_files_on_erda()
    # series_with_dim_errors = get_series_with_dim_error()

    base = r'Z:/IRE-DATA/CT/'
    folder = f'segmentations-colon/non-collapsed'
    directory_path = os.path.join(base, folder)
    # List to hold all file paths
    files_non_collapes = []
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            name = file.split("_sitk")[0]
            files_non_collapes.append(name)

    df['on_erda'] = False
    df['non_collapsed'] = False
    df['dimensions'] = None
    df['iop_list'] = None
    df['patient_position'] = None
    df['pixel_spacing'] = None
    df['slice_thickness'] = None
    df['image_position_patient'] = None

    # add additional data like:
    # "1.3.6.1.4.1.9328.50.81.16147424463095033147395563721259311594": {
    #  "dimensions": [ 512, 512, 557 ],
    #  "iop_list": [ -1.0, 0.0, 0.0, 0.0, -1.0, 0.0 ],
    #  "patient_position": "FFP",
    #  "pixel_spacing": [0.6640625, 0.6640625 ],
    #  "slice_thickness": 1.0,
    #  "image_position_patient": [ -135.66796875, -319.66796875, -560.0]
    #  }

    def get_additional_info(row):
        series_id = row['InstanceUID']
        name = row['name']
        # mark series whether they are on ERDA or not
        if series_id in erda_files.keys():
            row['on_erda'] = True

        # mark series whether they are non-collapsed
        if name in files_non_collapes:
            row['non_collapsed'] = True

        if series_id in additional_data.keys():
            row['dimensions'] = additional_data[series_id].get('dimensions', None)
            row['iop_list'] = additional_data[series_id].get('iop_list', None)
            row['patient_position'] = additional_data[series_id].get('patient_position', None)
            row['pixel_spacing'] = additional_data[series_id].get('pixel_spacing', None)
            row['slice_thickness'] = additional_data[series_id].get('slice_thickness', None)
            row['image_position_patient'] = additional_data[series_id].get('image_position_patient', None)
        else:
            row = find_missing_information(row)
        return row

    df = df.apply(get_additional_info, axis=1)
    columns_to_check = ['dimensions', 'iop_list', 'patient_position', 'pixel_spacing', 'slice_thickness', 'image_position_patient']
    has_none_values = df[columns_to_check].isnull().any()
    print(has_none_values)
    columns_to_delete = ['old_dicom_path', 'old_mha_path']  # Specify the columns you want to delete
    df = df.drop(columns=columns_to_delete)
    rename_dict = {
        'new_mha_path': 'mha_path',
        'new_dicom_path': 'dicom_path'
    }
    df = df.rename(columns=rename_dict)
    df.to_json('tcia-data/meta_data_df.json', orient='records', lines=True)


if __name__ == "__main__":
    # mapping_folder = os.path.join("tcia-data", "mapping")
    # os.makedirs(mapping_folder, exist_ok=True)
    #
    # create_meta_data_mapping()
    # create_sub_id()

    find_additional_information()


