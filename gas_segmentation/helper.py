import tempfile
import zipfile
import pandas as pd
import nibabel as nib
import gzip
import shutil
import os
import SimpleITK as sitk
import paraview.simple as pvs
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import itertools
import json
import time
import seaborn as sns
import jsonlines

base_folder = '/home/amin/PycharmProjects/WP-BIO'
erda_conv_folder = os.path.join('/home/amin/ucph-erda-home/IRE-DATA/CT', 'converted')


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


def find_first_index_1d(array):
    indices = np.where(array == 1)[0]  # Find all indices where the value is 1
    if indices.size > 0:
        return indices[0]  # Return the first index
    return None  # If 1 is not found


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


def add_patient_data(filename, volume=None, surface=None, time=None, threshold=None, neighbours=None, collapsed=None):
    # Patient data to be added
    patient_data = {
        "filename": filename,
        "volume": volume,
        "surface": surface,
        "time": time,
        "collapsed": collapsed,
    }

    file_path = f'tcia-data/meta-data/segmentation_details_{threshold}_{neighbours}.jsonl'
    file_path = os.path.join(base_folder, file_path)
    # Open file in append mode and write the JSON object as a line
    with open(file_path, 'a') as file:
        file.write(json.dumps(patient_data) + '\n')


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
    output_path = os.path.join(base_folder, "tcia-data", "plots", f"{new_file_name}_plt-org")
    visualize_image(sitk.GetArrayFromImage(image), output_path, title=f'{input_filepath} Original Image')

    # rotate image and visualize result
    image_rotated, meta_data, changed = orient_image(image, False, image.GetDirection())

    # img_data = sitk.GetArrayFromImage(image_rotated)
    # visualize_image(img_data, output_path, title=f'{input_filepath} Oriented Image')

    # Apply a global threshold to identify the luminal surface
    binary_image = image_rotated < threshold
    bi_img_data = sitk.GetArrayFromImage(binary_image)

    initial_point = find_initial_point(bi_img_data)

    # output_path2 = os.path.join(base_folder, "tcia-data", "plots", f"{new_file_name}_plt-bin")
    # visualize_image(bi_img_data, output_path2, title=f'{input_filepath} Binary Image - Thr: {threshold}', initial_point=initial_point)

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

    output_path3 = os.path.join(base_folder, "tcia-data", "plots", f"{new_file_name}_plt-seg")

    add_patient_data(new_file_name, volume, surface, elapsed_time, threshold, neighbours, collapsed)

    connected_segments = (connected_segments != 0).astype(np.uint8)

    # name the new file in a way it can be reconstructed how it was created
    filename = f"{new_file_name}.mha"

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

    # Compress the NIFTI file
    with open(source_path, 'rb') as f_in:
        with gzip.open(compressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"Compressed NIFTI file saved at: {compressed_path}")


def decompress_file(path, filename):
    #decompressed_nifti_path = f"{path}/{filename}"
    decompressed_nifti_path = os.path.join(path, filename)
    compressed_nifti_path = decompressed_nifti_path + '.gz'

    # Decompress the NIFTI file
    with gzip.open(compressed_nifti_path, 'rb') as f_in:
        with open(decompressed_nifti_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"Decompressed NIFTI file saved at: {decompressed_nifti_path}")


def visualize_slice(data, title=""):
    plt.imshow(data, cmap='gray')
    plt.title(title)
    plt.show()


def nifty_to_mha(filename, path):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(f"{path}{filename}.nii")
    image = reader.Execute()

    writer = sitk.ImageFileWriter()
    writer.SetFileName(f"{path}{filename}.mha")
    writer.Execute(image)


def visualize_histogram(data, title):
    plt.hist(data.ravel(), bins=2)
    plt.title(title)
    plt.xlabel('Scalar Value')
    plt.ylabel('Frequency')
    plt.show()


def create_mesh(source_path, target_path):
    # Load the segmentation image
    image = pvs.OpenDataFile(source_path)
    data_info = pvs.GetActiveSource()
    print(data_info)

    # Print available point and cell data arrays
    print("Point Data Arrays:")
    for array_name in data_info.PointData.keys():
        print(f"- {array_name}")

    print("Cell Data Arrays:")
    for array_name in data_info.CellData.keys():
        print(f"- {array_name}")

    # Apply the Contour filter to generate a 3D surface
    contour = pvs.Contour(Input=image)
    contour.ContourBy = ['MetaImage']  # Scalars_
    contour.Isosurfaces = [0.5]  # Set the contour value, adjust if needed
    contour.UpdatePipeline()

    # Check if the contour filter has output
    contour_output = contour.GetClientSideObject().GetOutput()
    if not contour_output.GetNumberOfPoints():
        raise RuntimeError("No valid geometry generated by the contour filter!")

    # Apply the Smooth filter
    smooth = pvs.Smooth(Input=contour)
    smooth.NumberofIterations = 50  # Adjust the number of iterations as needed
    smooth.Convergence = 0.0  # Adjust the convergence threshold as needed
    smooth.RelaxationFactor = 0.5
    smooth.UpdatePipeline()

    # Check if the smooth filter has output
    smooth_output = smooth.GetClientSideObject().GetOutput()
    if not smooth_output.GetNumberOfPoints():
        raise RuntimeError("No valid geometry generated by the smooth filter!")

    # # Apply the Decimate filter
    # decimate = pvs.Decimate(Input=smooth)
    # decimate.TargetReduction = 0.1  # Target reduction (0.5 = 50% reduction). Adjust as needed
    # decimate.UpdatePipeline()

    # # Check if the decimate filter has output
    # decimate_output = decimate.GetClientSideObject().GetOutput()
    # if not decimate_output.GetNumberOfPoints():
    #     raise RuntimeError("No valid geometry generated by the decimate filter!")

    # # Visualize the decimated data
    # decimate_display = pvs.Show(decimate)
    # pvs.Render()
    # time.sleep(1)

    # # Export the final mesh to a .ply file
    # pvs.SaveData(target_path, proxy=decimate, FileType='Binary')
    # print(f"Mesh exported to {target_path}")

    # # Clean up by deleting the data from the pipeline and hiding the display
    # pvs.Delete(decimate)

    # Export the final mesh to a .ply file
    pvs.SaveData(target_path, proxy=smooth, FileType='Binary')
    print(f"Mesh exported to {target_path}")
    pvs.Delete(smooth)


def show_ply_file(file):
    # Load the .ply file
    ply_data = pvs.OpenDataFile(file)
    print(f"Loaded {file}")

    # Display the loaded data
    display = pvs.Show(ply_data)

    # Set representation to 'Surface' for better visualization (optional)
    display.Representation = 'Surface'

    # Get the active view
    view = pvs.GetActiveViewOrCreate('RenderView')
    view.InteractionMode = '3D'

    # Get data bounds
    bounds = ply_data.GetDataInformation().GetBounds()
    center = [(bounds[1] + bounds[0]) / 2, (bounds[3] + bounds[2]) / 2, (bounds[5] + bounds[4]) / 2]
    max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

    # Zoom out factor
    zoom_out_factor = 2

    # List of camera positions and focal points
    camera_positions = [
        [center[0], center[1], bounds[5] + zoom_out_factor * max_dim],  # Top view
        [center[0] + zoom_out_factor * max_dim, center[1], center[2]],  # Front view (normal)
        [center[0] - zoom_out_factor * max_dim, center[1], center[2]]  # Back view
    ]
    focal_points = [
        center,  # Common focal point
        center,  # Common focal point
        center  # Common focal point
    ]
    view_up_vectors = [
        [0.0, 1.0, 0.0],  # Top view up direction
        [0.0, 0.0, 1.0],  # Front view up direction
        [0.0, 0.0, 1.0]  # Back view up direction
    ]
    # Iterate through camera positions
    for i in range(len(camera_positions)):
        view.CameraPosition = camera_positions[i]
        view.CameraFocalPoint = focal_points[i]
        view.CameraViewUp = view_up_vectors[i]
        pvs.Render()
        time.sleep(1)  # Adjust as needed

    print("Finished showing object from different angles.")

    # Clean up by deleting the data from the pipeline and hiding the display
    pvs.Delete(ply_data)


def investigate_nifty(filename, path):
    img = nib.load(f"{path}{filename}.nii")
    print(img.header)

    data = img.get_fdata()

    # Get the scalar range
    scalar_range = (np.min(data), np.max(data))
    print("NIfTI scalar range:", scalar_range)

    # Visualize histogram
    visualize_histogram(data, "NIfTI Scalar Values")

    # Visualize a slice from NIfTI data (optional)
    slice_index = data.shape[2] // 2  # Choose the middle slice index along the z-axis
    visualize_slice(data[:, :, slice_index], title=f"Slice {slice_index} from NIfTI data")


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


if __name__ == "__main__":
    #######################################################################################################
    plots_folder = os.path.join(base_folder, "tcia-data", "plots")
    os.makedirs(plots_folder, exist_ok=True)

    threshold = -800
    neighbours = 1
    files = ['sub286_pos-prone_scan-1_conv-sitk.mha.gz',
             'sub747_pos-prone_scan-1_conv-sitk.mha.gz',]

    for filename in files:
        subject = filename.split('_')[0]
        source_path = os.path.join(erda_conv_folder, subject, filename)
        #source_path = os.path.join(base_folder, "tcia-data", "converted", subject, filename)
        target_directory = os.path.join(base_folder, "tcia-data", "segmentations_colon")

        filepath_segmentation = create_segmentation(source_path, target_directory, filename, threshold=threshold,
                            neighbours=neighbours, coloured=False, only_thresholding=False)

        if filepath_segmentation:
            if os.path.isfile(filepath_segmentation):
                compress_file(filepath_segmentation, filepath_segmentation)
            else:
                print("No Segmentation File")

        ########################################################################################################
        # Example how to create meshes from an existing segmentation
        # segmented_filename = filepath_segmentation.split("/")[-1]
        # segmented_filename, ext1 = os.path.splitext(segmented_filename)
        #
        # # creates .ply file
        # mesh_filename = f"{segmented_filename}_pvs"
        # meshed_file = f"tcia-data/surfacemeshes/{mesh_filename}.ply"
        # create_mesh(filepath_segmentation, meshed_file)
        # compress_file(meshed_file, meshed_file)
        #
        # ###########################################################################################################
        # show_ply_file(f"tcia-data/surfacemeshes/{uuid}_sitk_{threshold}_{neighbours}_pvs.ply")

    print("DONE")

