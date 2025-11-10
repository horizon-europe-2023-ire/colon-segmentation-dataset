# Colon-Dataset
Contains scripts and tools relevant to produce the HQColon Dataset, a dataset of 435 human colons, segmented from Computed Tomography Colonography (CTC) obtained from the publicly available The Cancer Imaging Archive (TCIA).

# 1. Conda Environment
Setting up the Conda environment

``` Console
conda create -n IRE
conda activate IRE      
conda install numpy     
conda install scipy     
conda install matplotlib
conda install simpleitk  (conda install -c conda-forge simpleitk)
conda install paraview  
conda install request   
```

Install ITK-Snap or 3D Slicer as visualization tools.

* http://www.itksnap.org/pmwiki/pmwiki.php
* https://www.slicer.org/


# 2. Introduction
This section should give a quick overview how and what for to use the given code.

## Data

### Download Data from the Cancer Imaging Archive
The data can be downloaded using the download.py file. Running this script a folder tcia-data is created where the raw data as the converted data is saved as in the following structure presented.

Simply run the script to start downloading. This process can take several hours but can be stopped at any point, without losing any data that was already processed.

```
tcia-data
  ├── raw
  │   ├── subXXX
  │   │   └── raw DICOM content, naming subXXX.dcm
  │   ├── subXXXX
  │   │   └── ...
  ├── converted
  │   ├── subXXX
  │   │   └── converted content in mha format, naming subXXX_conv-sitk.mha
  │   ├── subXXXX
  │   │   └── ...
```
The dataset used is the CT COLONOGRAPHY | ACRIN 6664 from teh Cancer Imaging Archieve, publicily available at https://www.cancerimagingarchive.net/collection/ct-colonography/. 

In order to ensure quality of the dataset only dicom series where all dimensions are between 350 and 700 are considered. We assume normal images to have dimensions around (512,512,520)

## Create Segmentations
### Gas-filled Colon Segments
To segment the gas-filled segmentes of the colon we use the following two steps:
1. **Thresholding**: Using a threshold, all air-filled parts can be easily detected. This results in a segmentation including the colon, the small intestine, the lungs, and the surrounding areas.

2. **Growing Region / Neighbours**: To filter out only the colon, an initial point within the colon needs to be identified, which is used as the initial point for the region growing method. This way only points directly connected to the initial point in the colon will be considered.

These segmented colons are then saved as follows: 

```
tcia-data
  ├── segmentations_colon
  │   ├── subXXX
  │   │   └── segmented colons in mha format, naming subXXX_conv-slicer_threshold-XXX_neighbours-XXX.mha
  │   ├── subXXXX
  │   │   └── ...
```
### Fluid-filled Colon Segments
To segment the fluid-filled parts of the colon we used an interactive machine learning pipeline with RootPainter. This repository contains the pre-processing steps to produce the dataset for RootPainter, and the post processing steps to refine the predictions produced by RootPainter and combine the gas-filled segments with the fluid-filled segments, to produce full colon segmentations. The RootPainter project is available here: https://osf.io/8tkpm/.

## Create Meshes
In the last step, the segmented `.mha` file is transformed into meshes to a `.ply` file.

```
tcia-data
  ├── surfacemeshes
  │   ├── subXXX
  │   │   └── surface meshes in ply format, naming subXXX_conv-slicer_threshold-XXX_neighbours-XXX_converted_method.ply
  │   ├── subXXXX
  │   │   └── ...
```









