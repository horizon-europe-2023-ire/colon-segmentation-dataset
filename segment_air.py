import tempfile
import zipfile
import gzip
import shutil
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import itertools
import time
from pathlib import Path
import pandas as pd


base_folder = Path(__file__).resolve().parent


def find_connected_areas(image, initial_point, neighbours=1):
    """
    Given an initial point which is part of a segmented area, find all points that ly within the same area of segmented
    points.
    :param image: image data
    :param initial_point: point to initialise area of interest
    :param neighbours: 1: only direct neighbours count or 2: also indirect neighbours count
    :return: segmentation where area of points connected to the initial points is marked
    """
    # Initialize the connected_segments array with zeros
    initial_point = [int(i) for i in initial_point]
    connected_segments = np.zeros_like(image, dtype=np.int32)

    # Dimensions of the image
    depth, height, width = image.shape

    # Direction vectors for 6-connectivity in 3D (left, right, up, down, forward, backward)
    assert neighbours == 1 or neighbours == 2

    if neighbours == 2:
        directions = list(itertools.product([-2, -1, 0, 1, 2], repeat=3))

    elif neighbours == 1:
        directions = list(itertools.product([-1, 0, 1], repeat=3))

    def is_within_bounds(z, y, x):
        return 0 <= z < depth and 0 <= y < height and 0 <= x < width

    # Ensure the initial point is within bounds and is a part of the segment
    if not is_within_bounds(*initial_point) or image[initial_point[0], initial_point[1], initial_point[2]] != 1:
        raise ValueError("Initial point is out of bounds or not part of the segment.")

    queue = deque([initial_point])
    volume = 1
    surface = 0
    last_point = initial_point
    connected_segments[initial_point[0], initial_point[1], initial_point[2]] = volume
    while queue:
        z, y, x = queue.popleft()
        part_of_surface = False
        for dz, dy, dx in directions:
            nz, ny, nx = z + dz, y + dy, x + dx
            # if connected_segments is 0 it is not segmented yet
            # if connected_segment is -1 it is right next to a segmented voxel
            # if connected_segment is bigger than 1 then it is part of the segmented area
            if is_within_bounds(nz, ny, nx) and connected_segments[nz, ny, nx] == 0 and image[nz, ny, nx] == 1:
                volume += 1
                connected_segments[nz, ny, nx] = volume
                queue.append([nz, ny, nx])
                last_point = [nz, ny, nx]
            elif is_within_bounds(nz, ny, nx) and image[nz, ny, nx] == 0:
                # here we identify the outer bound of the connected segments
                part_of_surface = True

        if part_of_surface:
            surface += 1

    connected_segments = connected_segments / volume # to have values between 0 and 1

    return connected_segments, surface, volume, last_point


def find_first_index_2d(array):
    indices = np.argwhere(array == 1)  # Find all indices where the value is 1
    if indices.size > 0:
        return indices[0]  # Return the first index (row, column)
    return None  # If 1 is not found


def find_initial_point(data):
    """
    In this method an initial point for the colon is guessed. Using the center of the width and depth (of the axial plane)
    we define a square of 100 pixels around the center of the axial plane. Then the axial plane is moved
    along the x-axis from the bottom up and foucus only on the area whithin the square. As soon as we found
    the first segmented pixel, we use this as initial point. We only consider squares on the x-axis range between 50 and 150.
    This is to avoid any points belonging to the surrounding instead of the colon.
    :param data: includes the 3-dimensional image data.
    :return: the initial point if one is found.
    """
    # Dimensions of the image
    height, depth, width = data.shape

    mid_width = int(width / 2)
    mid_depth = int(depth / 2)

    # Slice the data to focus on the middle region
    possible_initial_points = data[50:250, mid_depth - 50:mid_depth + 50, mid_width]

    for i in range(possible_initial_points.shape[0]):
        point = find_first_index_2d(possible_initial_points[i, :])
        if point is not None:
            # Adjust point to match the original data's coordinates
            return [50 + i, mid_depth - 50 + point[0], mid_width]

    return None


def create_segmentation(input_filepath, output_directory, filepath, threshold=-800, neighbours=1, coloured=False,
                        only_thresholding=False):
    """
    Main method to create segmentation for given file. Additionally, the segmentation is plotted and saved.
    An initial point in the colon is guessed and all neighbours are searched for to identify the colon only
    to remove surrounding and small intestine from the segmentation.
    :param input_filepath: path of input file (the zip folder containing the mha file)
    :param output_directory: directory where to save segmentation
    :param threshold: what threshold to use
    :param neighbours: what distance is considered neighbour. Possible values: 1, 2
    :param coloured: Additional colours to visualize how and in what order neighbours are found
    :param only_thresholding: skips the part after thresholding
    :return: file path of segmentation
    """

    filename = filepath.replace(".mha.gz", '')
    base_name = filepath.replace("_conv-sitk.mha.gz", '')  # sub044_pos-prone_scan-1

    if input_filepath.endswith('.zip'):
        # Extract the .mha file from the ZIP archive
        with zipfile.ZipFile(input_filepath, 'r') as zip_file:
            # Find the .mha file within the zip archive
            for zip_info in zip_file.infolist():
                if zip_info.filename.endswith(".mha"):
                    # Read the .mha file from the archive
                    with zip_file.open(zip_info) as mha_file:
                        # Use a temporary file to read the MHA content
                        with tempfile.NamedTemporaryFile(suffix=".mha", delete=False) as temp_mha_file:
                            # Write the contents of the MHA file to the temporary file
                            temp_mha_file.write(mha_file.read())
                            temp_mha_filepath = temp_mha_file.name
        try:
            image = sitk.ReadImage(temp_mha_filepath)  # for linux?
        except Exception as e:
            image = sitk.ReadImage(temp_mha_file)  # for windows?
            os.remove(temp_mha_filepath)
    elif input_filepath.endswith('.gz'):
        # Handle the .gz file extraction
        with gzip.open(input_filepath, 'rb') as gz_file:
            # Create a temporary file to write the extracted content
            with tempfile.NamedTemporaryFile(suffix=".mha", delete=False) as temp_mha_file:
                temp_mha_file.write(gz_file.read())
                temp_mha_filepath = temp_mha_file.name
        try:
            image = sitk.ReadImage(temp_mha_filepath)  # for linux?
        except Exception as e:
            image = sitk.ReadImage(temp_mha_file)  # for windows?
            os.remove(temp_mha_filepath)
    elif input_filepath.endswith('.mha'):
        try:
            image = sitk.ReadImage(input_filepath)  # for linux?
        except Exception as e:
            raise ValueError(f"Tried open a .mha file, failed.")  # for windows?
    else:
        raise ValueError("Unsupported file format. Please provide a .zip or .gz file.")

    dimension = sitk.GetArrayFromImage(image).shape
    if not all(dim > 350 for dim in dimension) and all(dim < 700 for dim in dimension):
        print(f"Invalid dimension for: {base_name}")
        return None

    # Visualize the original image
    if threshold < 0:
        thr_str = f"n{abs(threshold)}"
    else:
        thr_str = str(threshold)
    new_file_name = f"{filename}_thr-{thr_str}_nei-{neighbours}"
    output_path = os.path.join(base_folder, "data", "air-plots", f"{new_file_name}_plt-org")
    visualize_image(sitk.GetArrayFromImage(image), output_path, title=f'{input_filepath} Original Image')

    # rotate image and visualize result
    image_rotated, meta_data, changed = orient_image(image, False, image.GetDirection())

    # Apply a global threshold to identify the luminal surface
    binary_image = image_rotated < threshold
    bi_img_data = sitk.GetArrayFromImage(binary_image)

    initial_point = find_initial_point(bi_img_data)

    if initial_point is None:
        print(f"Initial Point not found for file: {input_filepath}")
        return None

    if only_thresholding:
        print(f"Only thresholding")
        return None

    start_time = time.time()
    connected_segments, surface, volume, last_point = find_connected_areas(bi_img_data, initial_point, neighbours=neighbours)
    end_time = time.time()
    elapsed_time = end_time - start_time

    collapsed = False

    if elapsed_time < 100:
        print(f"Colon is probably collapsed for: {new_file_name}")
        collapsed = True

    output_path3 = os.path.join(base_folder, "data", "air-plots", f"{new_file_name}_plt-seg")

    connected_segments = (connected_segments != 0).astype(np.uint8)
    filename = f"{new_file_name}.mha"  # name the new file in a way it can be reconstructed how it was created

    # Save connected_segments as .mha file
    connected_segments_img = sitk.GetImageFromArray(connected_segments)

    if changed:
        connected_segments_img, _, __ = orient_image(connected_segments_img, True, image.GetDirection())
    connected_segments_img.CopyInformation(image)
    sitk.WriteImage(connected_segments_img, os.path.join(output_directory, filename))

    visualize_image(sitk.GetArrayFromImage(connected_segments_img), output_path3,
                    title=f'{input_filepath} Thr: {threshold} Time: {elapsed_time}',
                    initial_point=initial_point, end_point=last_point)

    print(f"successfully segmented {new_file_name}")

    return os.path.join(output_directory, filename)


def show_plot(ax, data, title="", points=None, cmap="tab20b"):
    ax.imshow(data, cmap=cmap)
    ax.set_title(title)
    return ax


def visualize_image(data, file_path, title='', initial_point=None, end_point=None, cmap="tab20b", show=False):
    # flip the z axis so the lungs are shown on top and the rectum on the bottom.
    data = np.flip(data, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0] = show_plot(axes[0], data[-70, :, :], title="Axial", cmap=cmap)
    axes[1] = show_plot(axes[1], data[:, data.shape[1] // 2, :], title="Coronal", cmap=cmap)
    axes[2] = show_plot(axes[2], data[:, :, data.shape[2] // 2], title="Sagital", cmap=cmap)

    # Convert SimpleITK image to a NumPy array for visualization
    if initial_point is not None:
        i = data.shape[0] - initial_point[0]
        axes[0].scatter(initial_point[1], initial_point[2], c='red')
        axes[1].scatter(initial_point[2], i, c='red')
        axes[2].scatter(initial_point[1], i, c='red')
    if end_point is not None:
        i = data.shape[0] - end_point[0]
        axes[0].scatter(end_point[1], end_point[2], c='white')
        axes[1].scatter(end_point[2], i, c='white')
        axes[2].scatter(end_point[1], i, c='white')

    plt.suptitle(title)
    plt.savefig(os.path.join(f"{file_path}.png"), format='png')

    if show:
        plt.show()
    plt.close()


def delete_file(file_path):
    try:
        os.remove(file_path)  # Delete the file
        print(f"File '{file_path}' deleted successfully.")
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except PermissionError:
        print(f"Permission denied to delete the file '{file_path}'.")
    except Exception as e:
        print(f"An error occurred while deleting the file: {e}")


def compress_file(source_path, target_path):
    compressed_path = target_path + '.gz'

    with open(source_path, 'rb') as f_in:
        with gzip.open(compressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # print(f"Compressed file saved at: {compressed_path}")


def transpose(img, direction):
    tmp = direction.copy()
    if direction[0] != 0:
        img = np.transpose(img, (0, 2, 1))
        direction[3:6] = tmp[6:]
        direction[6:] = tmp[3:6]
        return img, direction
    elif direction[4] != 0:
        img = np.transpose(img, (2, 1, 0))
        direction[:3] = tmp[6:]
        direction[6:] = tmp[:3]
        return img, direction
    elif direction[8] != 0:
        img = np.transpose(img, (1, 0, 2))
        direction[:3] = tmp[3:6]
        direction[3:6] = tmp[:3]
        return img, direction
    else:
        print("No orientation detected!")
        return img, direction


def flip_img(img, direction):
    matrix_as_numpy = np.array(direction)
    matrix_as_numpy = np.array(np.where(matrix_as_numpy == -1))[0]
    if 0 in matrix_as_numpy:
        img = np.flip(img, axis=2)
    if 4 in matrix_as_numpy:
        img = np.flip(img, axis=1)
    if 8 in matrix_as_numpy:
        img = np.flip(img, axis=0)
    return img


def orient_image(image, revert, direction):
    """
    This method makes sure all images of the dataset are oriented in the same way: LR - AP - IS
    :param image: the image data in sitk format
    :param revert: boolean if we reorient the first time or do the revert action
    :param direction: the direction of the orientation
    :return:
    """
    direction = list(direction)
    image_array = sitk.GetArrayFromImage(image)
    meta_data = {"spacing": image.GetSpacing(),
                 "origin": image.GetOrigin(),
                 "direction": image.GetDirection(),
                 "dimension": image.GetDimension()
                 }

    if not revert:
        change = False

        # check if we have to transpose image
        if direction[0] == 0 or direction[4] == 0 or direction[8] == 0:
            change = True
            image_array, direction = transpose(image_array, direction)

        # check if we have to flip or transpose image
        tmp = np.array(direction) == np.abs(direction)
        if not tmp.all():
            change = True
            image_array = flip_img(image_array, direction)

        # update metadata
        if change:
            rotated_image = sitk.GetImageFromArray(image_array)
            return rotated_image, meta_data, True

    else:
        # check if we have to flip or transpose image
        tmp = np.array(direction) == np.abs(direction)
        if not tmp.all():
            image_array = flip_img(image_array, direction)

        # check if we have to transpose image
        if direction[0] == 0 or direction[4] == 0 or direction[8] == 0:
            image_array, direction = transpose(image_array, direction)

        rotated_image = sitk.GetImageFromArray(image_array)
        return rotated_image, meta_data, True

    return image, meta_data, False


def list_gz_files(folder_path):
    gz_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.gz'):
                base_name = os.path.basename(file)
                base_name = '_'.join(base_name.split('_')[:4])
                gz_files.append(base_name)
    return gz_files


def segment_air():
    converted_folder = os.path.join(base_folder, "data", "converted")
    segmentations_colon_folder = os.path.join(base_folder, "data", "segmentations-air")
    os.makedirs(segmentations_colon_folder, exist_ok=True)
    os.makedirs(os.path.join(base_folder, "data", "air-plots"), exist_ok=True)

    # Check if the 'converted' folder exists
    if not os.path.isdir(converted_folder):
        raise ValueError(f"The directory {converted_folder} does not exist. No data to segment.")

    # List all files in the 'converted' folder
    # filenames: sub002_pos-prone_scan-1_conv-sitk_thr-n800_nei-1.mha.gz
    existing_seg_files = list_gz_files(segmentations_colon_folder)
    filenames = list_gz_files(converted_folder)

    threshold = -800
    neighbours = 1

    for file in filenames:
        filename = file.split('conv')[0][:-1]
        if filename in existing_seg_files:
            # indicated that that file was already processed, skip to next file
            print(f"Skipping {filename}")
            continue

        subject = file.split('_')[0]

        # source_path = f"{converted_folder}/{subject}/{file}"
        source_path = os.path.join(base_folder, 'data', 'converted', subject, file)
        target_directory = os.path.join(segmentations_colon_folder, subject)
        os.makedirs(target_directory, exist_ok=True)

        existing_segmentations = os.listdir(target_directory)
        is_present = any(file in s for s in existing_segmentations)
        if is_present:
            print(f"Already segmented: {file}")
            continue

        print(f"Start to segment: {file}")
        segmented_filepath = create_segmentation(source_path, target_directory, file, threshold=threshold,
                                                 neighbours=neighbours,
                                                 coloured=True, only_thresholding=False)

        if segmented_filepath is None:
            continue

        compress_file(segmented_filepath, segmented_filepath)
        delete_file(segmented_filepath)
