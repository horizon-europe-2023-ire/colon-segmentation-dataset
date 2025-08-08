from segment_air import segment_air
from download import download, convert_manually_downloaded

"""
This script can be used to download the CT COLONOGRAPHY dataset from TCIA, convert it into .mha files and then segment 
the air using a combination of thresholding and connectivity. 

To achieve this a heuristic approach to set a seed point in the anus is used.

Instead of downloading the entire dataset (which might take many hours), you can define an array of series ids you are
interested in as shown below.
"""


if __name__ == "__main__":

    files = ["1.3.6.1.4.1.9328.50.4.850207",
             "1.3.6.1.4.1.9328.50.4.849142",
             "1.3.6.1.4.1.9328.50.4.692452",
             "1.3.6.1.4.1.9328.50.4.691902",
             "1.3.6.1.4.1.9328.50.4.210670",
             "1.3.6.1.4.1.9328.50.4.210003",
             "1.3.6.1.4.1.9328.50.4.113397",
             "1.3.6.1.4.1.9328.50.4.112806",
             "1.3.6.1.4.1.9328.50.4.531869",
             "1.3.6.1.4.1.9328.50.4.531406",
             "1.3.6.1.4.1.9328.50.4.389235",
             "1.3.6.1.4.1.9328.50.4.434283",
             "1.3.6.1.4.1.9328.50.81.202700814695273699097907559082771609192"
             ]
    download(files)

    # convert_manually_downloaded()

    segment_air()

