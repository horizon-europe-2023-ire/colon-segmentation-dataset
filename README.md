# 🧬 Colon-Segmentation-Dataset

This repository contains scripts and tools used to produce the **HQColon Dataset** — a dataset of **435 human colons** segmented from **Computed Tomography Colonography (CTC)** scans obtained from the publicly available [**Cancer Imaging Archive (TCIA)**](https://www.cancerimagingarchive.net/collection/ct-colonography/).

The pipeline covers the full process:  
1. Downloading and converting CT data  
2. Segmenting colon **air**, **fluid**

---

## ⚙️ 1. Conda Environment Setup

To reproduce the segmentation pipelines, first set up your conda environment:

```console
conda create -n IRE python=3.10
conda activate IRE

conda install numpy
conda install scipy
conda install matplotlib
conda install -c conda-forge simpleitk
conda install paraview
conda install requests
```

For visualization, install **ITK-Snap** or **3D Slicer**:

- 🧠 [ITK-Snap](http://www.itksnap.org/pmwiki/pmwiki.php)
- 🧩 [3D Slicer](https://www.slicer.org/)

---

## 🧭 2. Repository Overview

```
.
├── download.py                         # Downloads and converts TCIA CT Colonography data
├── gas_segmentation/                   # Colon air segmentation pipeline
│   ├── run_pipeline.py
│   ├── segment_gas.py
│   └── ...
├── fluid_segmentation/                 # RootPainter-based colon fluid segmentation
│   ├── generate_rootpainter_dataset.py
│   ├── match_png_to_volumes.py
│   ├── resample_segmentation.py
│   └── ...
├── totalsegmentator_pipeline/          # TotalSegmentator organ segmentation
│   ├── create_image_paths.py
│   ├── filter_unprocessed_images.py
│   ├── split_paths_into_batches.py
│   ├── totalsegmentator_oneimage.py
│   ├── totalsegmentator_batchimages.py
│   └── ...
└── data/
    ├── converted/
    ├── segmentation-gas/
    ├── segmentations_totalsegmentator/
    ├── segmentations-air/
    └── ...
```

---

## 🧩 3. Data Overview

### Download Data from The Cancer Imaging Archive (TCIA)

The dataset used is **CT COLONOGRAPHY | ACRIN 6664**, publicly available from TCIA.

Run the following script to download and convert the dataset:

```bash
python download.py
```

This will create a `data/` folder containing the following structure:

```
data/
  ├── raw/
  │   ├── subXXX/
  │   │   └── raw DICOM content (*.dcm)
  ├── converted/
  │   ├── subXXX/
  │   │   └── converted image (*.mha)
```

📏 **Note:**  
To ensure high-quality data, only DICOM series where all dimensions are between **350 and 700** are included  
(typical CT shape ≈ `(512, 512, 520)`).

---

## 💨 4. Gas-Filled Colon Segmentation

Located in: `gas_segmentation/`

This step isolates **air-filled segments** of the colon.

### Method

1. **Thresholding** – Detects air regions (e.g., below -800 HU).  
2. **Region Growing** – Starting from an automatically detected seed point (anus area), keeps only connected colon regions.

### Run

```bash
python gas_segmentation/run_pipeline.py
```

### Output Structure

```
data/
  ├── segmentation-gas/
  │   ├── subXXX/
  │   │   └── subXXX_conv-sitk_thr-n800_nei-1.mha
```

---

## 💧 5. Fluid-Filled Colon Segmentation

Located in: `fluid_segmentation/`

This pipeline identifies **fluid-filled** colon regions using **RootPainter** (interactive machine learning).

### Steps

1. **Generate PNG dataset**
   ```bash
   python fluid_segmentation/generate_rootpainter_dataset.py <input_dir> <output_dir>
   ```

2. **Annotate and run inference** with [RootPainter](https://osf.io/8tkpm/).

3. **Reconstruct 3D volumes from PNG masks**
   ```bash
   python fluid_segmentation/match_png_to_volumes.py
   ```

4. **Resample segmentations** to match original scan geometry
   ```bash
   python fluid_segmentation/resample_segmentation.py
   ```

5. **Extract only fluid regions**
   ```bash
   python fluid_segmentation/subtract_labelmaps.py
   ```

---

## 🧠 6. TotalSegmentator – Multi-Organ Segmentation

Located in: `totalsegmentator_pipeline/`

Uses [**TotalSegmentator**](https://github.com/wasserth/TotalSegmentator) to automatically segment major organs and body structures.

### Install TotalSegmentator

```bash
pip install totalsegmentator
```

### Run the Full Pipeline

1. **Generate image paths**
   ```bash
   python totalsegmentator_pipeline/create_image_paths.py
   ```

2. **Filter unprocessed images**
   ```bash
   python totalsegmentator_pipeline/filter_unprocessed_images.py
   ```

3. **Split into batches**
   ```bash
   python totalsegmentator_pipeline/split_paths_into_batches.py
   ```

4. **Run segmentation**
   ```bash
   python totalsegmentator_pipeline/totalsegmentator_batchimages.py image_paths.txt
   ```

### Output Example

```
data/segmentations_totalsegmentator/
└── sub001_pos-supine_scan-1_conv-sitk/
    ├── sub001_pos-supine_scan-1_conv-sitk_totalseg-liver.mha.gz
    ├── sub001_pos-supine_scan-1_conv-sitk_totalseg-heart.mha.gz
    └── sub001_pos-supine_scan-1_conv-sitk_totalseg-lung_left.mha.gz
```

---

## 🧪 8. Example Full Workflow

```bash
# 1. Download data
python download.py

# 2. Segment air regions
python gas_segmentation/run_pipeline.py

# 3. Generate RootPainter input dataset
python fluid_segmentation/generate_rootpainter_dataset.py data/converted data/rootpainter_input

# 4. Run RootPainter manually, then reconstruct results
python fluid_segmentation/match_png_to_volumes.py

# 5. Segment organs
python totalsegmentator_pipeline/totalsegmentator_batchimages.py image_paths.txt
```

---

## 🧩 Dependencies

Install all required packages (Python ≥ 3.9):

```bash
pip install numpy scipy matplotlib SimpleITK totalsegmentator pillow tqdm pydicom
```

Optional (for visualization):

- [ITK-Snap](http://www.itksnap.org/pmwiki/pmwiki.php)
- [3D Slicer](https://www.slicer.org/)
- [Paraview](https://www.paraview.org/)

---

## 📬 Contact

For questions, feedback, or collaboration:

📧 [martina.finocchiaro.mf@gmail.com](mailto:martina.finocchiaro.mf@gmail.com])  

---

✅ **Summary**  
This repository provides a **complete CT Colonography segmentation framework** — from raw DICOM downloads to 3D organ and colon mask generation — using a combination of traditional image processing, interactive labeling, and state-of-the-art deep learning segmentation.
