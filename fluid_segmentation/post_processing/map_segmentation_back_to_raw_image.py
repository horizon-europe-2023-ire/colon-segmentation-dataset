import SimpleITK as sitk
import os
import re

"""
    Remap segmentations based on the original image size (in case they had a different size/origin/direction)
"""

def find_matching_original(segmentation_filename, original_folder):
    """
    Find the original image that corresponds to the segmentation file based on the three-digit number after 'colon_'.
    """
    match = re.search(r'colon_(\d{3})', segmentation_filename)
    if not match:
        print(f"Warning: Could not extract number from {segmentation_filename}, skipping...")
        return None

    number = match.group(1)

    # Search for a file in the original folder that contains this number
    for filename in os.listdir(original_folder):
        if number in filename and filename.endswith(".mha"):
            return os.path.join(original_folder, filename)
    
    print(f"Warning: No matching original image found for {segmentation_filename} (number {number}), skipping...")
    return None

def resample_segmentation_folder(segmentation_folder, original_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(segmentation_folder):
        if filename.endswith(".mha"):
            segmentation_path = os.path.join(segmentation_folder, filename)
            original_image_path = find_matching_original(filename, original_folder)

            if not original_image_path:
                continue  # Skip if no match found

            # Generate the output filename by replacing 'v2' with 'v3'
            output_filename = filename.replace("v2", "v3")
            output_path = os.path.join(output_folder, output_filename)

            # Load images
            segmentation = sitk.ReadImage(segmentation_path)
            original_image = sitk.ReadImage(original_image_path)

            # Get metadata from original image
            original_size = original_image.GetSize()
            original_spacing = original_image.GetSpacing()
            original_origin = original_image.GetOrigin()
            original_direction = original_image.GetDirection()

            # Define resampling filter
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(original_spacing)
            resampler.SetSize(original_size)
            resampler.SetOutputOrigin(original_origin)
            resampler.SetOutputDirection(original_direction)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Preserve segmentation labels

            # Apply resampling
            resampled_segmentation = resampler.Execute(segmentation)

            # Save the resampled segmentation
            sitk.WriteImage(resampled_segmentation, output_path)
            print(f"Processed: {filename} -> {output_filename}")


# usage
segmentation_folder = "/home/martina/Dataset/rootpainter_inference_mask_mha/postprocessing/to_modify_for_dim"
original_folder = "/home/martina/Dataset/rootpainter_inference_mask_mha/postprocessing/orginal_for_modification"
output_folder = "/home/martina/Dataset/rootpainter_inference_mask_mha/postprocessing/modified_for_dim"

resample_segmentation_folder(segmentation_folder, original_folder, output_folder)