# 🧠 TotalSegmentator Batch Processing Pipeline

This pipeline automates **organ segmentation** of 3D CT images using [**TotalSegmentator**](https://github.com/wasserth/TotalSegmentator).  
It processes all input images stored in the `data/` directory, managing input/output files and supporting both **batch** and **parallel** execution.

---

## 🗂️ Dataset and Directory Layout

Your data directory should follow this structure:

```
data/
├── converted/                      # Preprocessed input images (.mha.gz)
└── segmentations_totalsegmentator/ # TotalSegmentator output (.mha.gz per organ)
```

Each output segmentation file follows the naming convention:

```
<input_name>_totalseg-<organ>.mha.gz
```

**Example:**

If the input file is:

```
sub001_pos-supine_scan-1_conv-sitk.mha.gz
```

Then the left lung segmentation will be named:

```
sub001_pos-supine_scan-1_conv-sitk_totalseg-lung_left.mha.gz
```

This standardized format allows easy programmatic access to specific organ masks.

---

## ⚙️ Pipeline Overview

### **Step 1 – Prepare Image Paths**

#### 1️⃣ Generate image paths
Create a list of all `.mha.gz` (or `.mha.zip`) files in `data/converted`:

```bash
python create_image_paths.py
```

→ Produces: `image_paths.txt`

#### 2️⃣ Filter unprocessed images
Exclude already segmented images to avoid redundant processing:

```bash
python filter_unprocessed_images.py
```

→ Produces: `image_paths_filtered.txt`

#### 3️⃣ Split image paths into batches
Divide large image lists into smaller, manageable batch files:

```bash
python split_paths_into_batches.py
```

→ Creates: `batch_paths/batch_1.txt`, `batch_paths/batch_2.txt`, etc.

---

### **Step 2 – Run Batch Segmentation**

#### 🖥️ Option A — Multi-GPU / Cluster Mode

If your compute environment supports multiple nodes with shared storage, you can launch batch jobs in parallel:

```bash
python totalsegmentator_batchimages.py batch_paths/batch_1.txt
```

Each batch file runs independently, allowing parallel segmentation across nodes.

#### 💻 Option B — Single-Machine Mode

If shared mounting or distributed jobs are unreliable, you can run everything locally in a single session.

1. **Start an interactive GPU session:**
   ```bash
   srun -p gpu --gres=gpu:a100:1 --pty bash
   ```

2. **Run the full pipeline:**
   ```bash
   python totalsegmentator_batchimages.py image_paths.txt
   ```

> ⏱️ For ~1600 CT volumes on a single NVIDIA A100, processing typically takes **3–4 days**.

---

## 📂 Output Structure

Each input image gets its own folder under `data/segmentations_totalsegmentator/`, containing multiple organ-specific segmentation masks.

**Example:**

```
data/segmentations_totalsegmentator/
└── sub001_pos-supine_scan-1_conv-sitk/
    ├── sub001_pos-supine_scan-1_conv-sitk_totalseg-liver.mha.gz
    ├── sub001_pos-supine_scan-1_conv-sitk_totalseg-heart.mha.gz
    ├── sub001_pos-supine_scan-1_conv-sitk_totalseg-lung_left.mha.gz
    └── sub001_pos-supine_scan-1_conv-sitk_totalseg-lung_right.mha.gz
```

---

## 📜 Script Summary

| Script | Purpose |
|--------|----------|
| **`create_image_paths.py`** | Scans `data/converted` and generates `image_paths.txt` with paths to all input images. |
| **`filter_unprocessed_images.py`** | Creates `image_paths_filtered.txt` with paths to images that haven’t been segmented yet. |
| **`split_paths_into_batches.py`** | Divides the list of image paths into smaller batch files for distributed processing. |
| **`totalsegmentator_batchimages.py`** | Executes segmentation in parallel for each image listed in a batch file. |
| **`totalsegmentator_oneimage.py`** | Core worker: decompresses `.mha.gz`, runs TotalSegmentator, converts outputs back to `.mha.gz`, and renames them consistently. |

---

## 🧩 Notes & Tips

- All scripts assume relative paths starting from the project root.
- The segmentation output structure mirrors the input directory hierarchy.
- The default parallelization uses **4 processes**; adjust as needed in `totalsegmentator_batchimages.py`.
- Any warnings or errors from TotalSegmentator runs are logged in:
  ```
  warnings.log
  ```
- Ensure dependencies are installed:

```bash
pip install totalsegmentator SimpleITK numpy
```

---

## 📬 Contact

For questions, feedback, or collaboration:

**Martina and Viktor**  
📧 [martina.finocchiaro.mf@gmail.com](mailto:martina.finocchiaro.mf@gmail.com])  
📧 [vikkimar03@gmail.com](mailto:vikkimar03@gmail.com)

---

✅ **Summary:**  
This README explains how to prepare image paths, batch them, and run the TotalSegmentator pipeline efficiently — either on a single GPU or across multiple machines — with full automation of file management and format conversion.
