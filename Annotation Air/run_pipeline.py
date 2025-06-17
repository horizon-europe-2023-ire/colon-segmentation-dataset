from segment_air import segment_air
from download import download

"""
This script can be used to download the CT COLONOGRAPHY dataset from TCIA, convert it into .mha files and then segment 
the air using a combination of thresholding and connectivity. 

To achieve this a heuristic approach to set a seed point in the anus is used.

Instead of downloading the entire dataset (which might take many hours), you can define an array of series ids you are
interested in as shown below.
"""


if __name__ == "__main__":
    files = None
    files = ["1.3.6.1.4.1.9328.50.4.867016", "1.3.6.1.4.1.9328.50.4.693009", ]
    download(files)
    segment_air()

