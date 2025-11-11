"""
===============================================================================
 Script: run_pipeline.py
 Purpose:
     End-to-end pipeline to:
       1) Download CT Colonography data from TCIA,
       2) Convert downloaded DICOM series to .mha format, and
       3) Segment intra-luminal gas (air) in the colon.

 Description:
     - The script coordinates the data retrieval, conversion, and segmentation
       process using helper modules:
           • download.py — handles data fetching and DICOM-to-MHA conversion.
           • segment_gas.py — performs air segmentation using intensity
             thresholding and connected-component growth from a heuristic
             seed point near the anus.
     - By default, only a subset of SeriesInstanceUIDs is processed to keep
       runtime manageable, but you can easily switch to downloading the full
       dataset if desired.

 Workflow:
     1. Specify one or more TCIA SeriesInstanceUIDs in the `files` list.
     2. The script calls `download(files)` to fetch and convert the data into:
            ../data/converted/<subject>/<scan>.mha(.gz)
     3. Once data are prepared, `segment_gas()` is called to segment the
        intra-luminal air and save results under:
            ../data/segmentation-gas/
        along with PNG quality-control visualizations under:
            ../data/air-plots/

 Usage:
     python run_pipeline.py

 Customization:
     - To process the entire dataset (⚠ may take >24h and large disk space),
       set `files = None` inside the script before running.
     - If you have already manually downloaded data, you can skip the
       `download()` step and use:
           convert_manually_downloaded()
     - Ensure TCIA network access is available and the output folders exist
       or can be created under the project directory.

===============================================================================
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segment_gas import segment_gas
from download import download, convert_manually_downloaded

if __name__ == "__main__":

    files = ["1.3.6.1.4.1.9328.50.4.850207",
             "1.3.6.1.4.1.9328.50.4.849142",
             "1.3.6.1.4.1.9328.50.4.692452",
             "1.3.6.1.4.1.9328.50.4.691902"
             ]
    
    download(files)

    # convert_manually_downloaded()

    segment_gas()

