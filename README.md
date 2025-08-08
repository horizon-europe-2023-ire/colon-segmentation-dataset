# Colon-Dataset

## Metadata

Metadata can be found in the metadata.jsonl file. In the following we summarize all available columns:

- **name**  
  Identifier name for the scan, created by us. E.g. sub006_pos-prone_scan-1 where sub006 identifies the subject (with our own sub_id) pos-prone describes the patient position of this scan and scan-1 identifies the scan id, for cases where patients have multiple scans in the same position.

- **filename_segmentation**  
  Filename of the segmentation file related to the scan created by us based on "name". E.g. sub006_pos-prone_scan-1_conv-sitk_thr-n800_nei-1: conv-sitk describes us using sitk to convert dicom to mha, thr-n800 stands for using a threshold of -800 to create binary maps, and nei-1 says using neighbours that are 1 voxel away.

- **InstanceUID**  
  Unique identifier for the DICOM instance from TCIA.

- **dim**  
  Dimensions of the image volume (width, height, depth). (extracted from TCIA)

- **iop_list**  
  Image Orientation Patient list — direction cosines of the first row and first column. (extracted from TCIA)

- **patient_position**  
  Position of the patient during the scan (e.g., Head First Supine). For some rare cases we identified errors, in the annotated position, which we marked as Wrong. (extracted from TCIA)

- **pixel_spacing**  
  Physical distance between pixels in millimeters. (extracted from TCIA)

- **slice_thickness**  
  Thickness of each image slice in millimeters. (extracted from TCIA)

- **image_position_patient**  
  The (x, y, z) coordinates of the first pixel in the patient coordinate system. (extracted from TCIA)

- **patients_history**  
  Patient’s medical history notes (may be unavailable). (extracted from TCIA)

- **SubjectUID**  
  Unique identifier for the subject/patient from TICA. (extracted from TCIA)

- **patients_age**  
  Age of the patient in years. (extracted from TCIA)

- **slice_location**  
  Position of the slice along the axis perpendicular to the slices. (extracted from TCIA)

- **gender**  
  Patient’s gender. (extracted from TCIA)

- **new_sub_id**  
  Simplified new subject identifier created by us.

- **scan**  
  Scan number or identifier.

- **position**  
  Patient position during scan (prone or supine).

- **split**  
  Dataset split label, e.g., train, validation, test (relevant for training nnunet).

- **colon_name**  
  Name or identifier for the specific colon or region analyzed (relevant for training nnunet).


## Download TCIA Dataset
run the run_pipeline.py to download the entire TCIA dataset and convert dicom to mha files.

### Download though pipeline
To download instances or the entire TCIA CT Colonography dataset the code in download.py file is used. 
The main method to call is: "download(files)". Files is a list of series instances like: ["1.3.6.1.4.1.9328.50.4.850207", "1.3.6.1.4.1.9328.50.4.849142"] or None. If the list is not empty only those instances will be downloaded. If the list is empty or None, the entire CT Colonography dataset is downloaded.

### Download manually through manifest
Instead, one can manually download files from tcia. Important is that folders containing dicom images should be placed in a folder "data/raw_manually".
In the run_pipeline file use method: "convert_manually_downloaded" instead of download. 
IMPORTANT: Note that only for dicom files form the CT Colonography dataset this will work / only if the series uid can be found in the meta_data_df.json file.

'python run_pipeline.py'









