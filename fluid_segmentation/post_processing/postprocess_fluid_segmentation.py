"""
===============================================================================
 Script: postprocess_fluid_segmentation.py
 Purpose:
     Post-process fluid segmentations (from RootPainter) using the corresponding
     colon air-filled segmentations. Produces:
       - cleaned/attached fluid masks (.mha)
       - optional 3D surface meshes (.vtk) in RAI orientation

 Description:
     - Reads a fluid segmentation `.mha` file and its paired colon (air-filled)
       segmentation `.mha`.
     - Reorients arrays to a consistent RAI reference using the image's
       direction matrix, computes distance maps, filters/attaches components,
       fills holes and smooths masks.
     - Restores original orientation for saving the cleaned fluid mask.
     - Creates a VTK mesh from the (RAI) cleaned mask for visualization.

 Usage:
     python postprocess_fluid_segmentation.py \
         <input_fluid_mha_dir> <input_colon_mha_dir> \
         <output_fluid_mesh_dir> <output_fluid_mha_dir> [max_files]

 Example:
     python postprocess_fluid_segmentation.py \
         ../data/fluid_masks_mha/TestSet \
         ../data/colon_masks_mha/labelsTs \
         ../data/fluid_mesh/TestSet \
         ../data/fluid_masks_cleaned/TestSet \
         100

 Notes:
     - Fluid/colon files are matched by a filename prefix regex: (.*colon_\\d{3})
       found in the fluid filename, e.g., "patient1_colon_012_cleaned.mha"
       → matches colon "patient1_colon_012.mha".
     - Output files:
         <fluid_basename>_cleaned_attached.mha
         <fluid_basename>_cleaned_attached.vtk
===============================================================================
"""

from pathlib import Path
import re
import numpy as np
import vtk
from skimage import measure

import SimpleITK as sitk
import scipy.ndimage as ndimage
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.spatial import KDTree


def reorient_array_with_matrix_M(image_array: np.ndarray, direction, origin):
    """
    Reorient a 3D NumPy array (z, y, x) using a 3x3 direction matrix to a
    consistent RAI orientation (nearest-neighbor remapping, no interpolation).
    """
    # Transpose to (x, y, z) like ITK
    T_image_array = np.transpose(image_array, (2, 1, 0))
    x_size, y_size, z_size = T_image_array.shape
    T_image_array_RAI = np.zeros_like(T_image_array)

    # Grid of all voxel coordinates
    coords = np.mgrid[0:x_size, 0:y_size, 0:z_size]
    coords = coords.reshape(3, -1).astype(np.float32).T  # (N, 3)

    direction_matrix = np.array(direction).reshape(3, 3)
    new_coords = np.dot(coords, direction_matrix.T)
    new_coords = np.rint(new_coords).astype(np.int32)

    old_x = coords[:, 0].astype(np.int32)
    old_y = coords[:, 1].astype(np.int32)
    old_z = coords[:, 2].astype(np.int32)

    new_x = new_coords[:, 0]
    new_y = new_coords[:, 1]
    new_z = new_coords[:, 2]

    # Vectorized re-mapping; collisions resolve by last write
    # (Assumes direction matrix is a permutation/reflection; no bounds checks added)
    T_image_array_RAI[new_x, new_y, new_z] = T_image_array[old_x, old_y, old_z]

    return T_image_array_RAI


def revert_reorientation(image_array: np.ndarray, direction):
    """
    Invert the transformation applied by `reorient_array_with_matrix_M`
    to restore to original reference.
    """
    x_size, y_size, z_size = image_array.shape
    restored = np.zeros_like(image_array)

    coords = np.mgrid[0:x_size, 0:y_size, 0:z_size]
    coords = coords.reshape(3, -1).astype(np.float32).T  # (N, 3)

    direction_matrix = np.array(direction).reshape(3, 3)
    inv_direction_matrix = np.linalg.inv(direction_matrix)

    original_coords = np.dot(coords, inv_direction_matrix.T)
    original_coords = np.rint(original_coords).astype(np.int32)

    old_x = coords[:, 0].astype(np.int32)
    old_y = coords[:, 1].astype(np.int32)
    old_z = coords[:, 2].astype(np.int32)

    new_x = original_coords[:, 0]
    new_y = original_coords[:, 1]
    new_z = original_coords[:, 2]

    restored[new_x, new_y, new_z] = image_array[old_x, old_y, old_z]
    # Back to (z, y, x)
    restored = np.transpose(restored, (2, 1, 0))
    return restored


def get_colon_surface_mask(colon_array: np.ndarray) -> np.ndarray:
    """Compute a binary surface mask for the colon (erode-then-diff)."""
    eroded = binary_erosion(colon_array, structure=np.ones((3, 3, 3)))
    surface_mask = (colon_array == 1) & (eroded == 0)
    return surface_mask


def build_colon_surface_kdtree(colon_surface_mask: np.ndarray):
    """Build KD-tree of colon surface voxel coordinates."""
    surface_indices = np.argwhere(colon_surface_mask)  # (N, 3) as (X, Y, Z) in RAI array
    tree = KDTree(surface_indices)
    return tree, surface_indices


def clean_fluid_segmentation(colon_mha_path: Path, fluid_mha_path: Path):
    """
    Load colon (air-filled) and fluid .mha masks, clean and attach fluid to colon,
    return arrays in RAI orientation + metadata.
    """
    colon_image = sitk.ReadImage(str(colon_mha_path))  # (x, y, z)
    fluid_image = sitk.ReadImage(str(fluid_mha_path))  # (x, y, z)

    colon_array = sitk.GetArrayFromImage(colon_image)  # (z, y, x)
    fluid_array = sitk.GetArrayFromImage(fluid_image)  # (z, y, x)

    spacing = colon_image.GetSpacing()
    origin = colon_image.GetOrigin()
    direction = colon_image.GetDirection()

    metadata = {"spacing": spacing, "origin": origin, "direction": direction}

    if colon_array.shape != fluid_array.shape:
        raise ValueError("Colon and fluid masks must have the same dimensions.")

    # Reorient to RAI
    colon_array_RAI = reorient_array_with_matrix_M(colon_array, direction, origin)
    fluid_array_RAI = reorient_array_with_matrix_M(fluid_array, direction, origin)
    print("Arrays rotated to RAI.")

    # 1) Colon distance map
    colon_distance_map = sitk.SignedMaurerDistanceMap(
        sitk.GetImageFromArray(colon_array_RAI.astype(np.uint8)),
        insideIsPositive=False,
        useImageSpacing=True,
    )
    colon_distance_array = sitk.GetArrayFromImage(colon_distance_map)

    # 2) Connected components in fluid
    fluid_cc = sitk.ConnectedComponent(sitk.GetImageFromArray(fluid_array_RAI.astype(np.uint8)))
    fluid_cc_array = sitk.GetArrayFromImage(fluid_cc)
    labels = np.unique(fluid_cc_array)
    labels = labels[labels != 0]

    # 3) Keep components near colon and large enough
    distance_threshold = 2.0  # mm
    min_voxel_threshold = 2000
    new_fluid_array = np.zeros_like(fluid_array_RAI, dtype=np.uint8)

    for lbl in labels:
        component_mask = (fluid_cc_array == lbl)
        component_size = int(component_mask.sum())
        component_distances = colon_distance_array[component_mask]
        # filter negative distances
        component_distances = component_distances[component_distances > 0]
        if component_distances.size == 0:
            continue
        min_distance = float(component_distances.min())
        if component_size >= min_voxel_threshold and min_distance <= distance_threshold:
            new_fluid_array[component_mask] = 1

    print("Fluid cleaned - pass 1")

    # 4) CC again after cleaning
    fluid_cc_clean = sitk.ConnectedComponent(sitk.GetImageFromArray(new_fluid_array))
    fluid_cc_clean_array = sitk.GetArrayFromImage(fluid_cc_clean)

    # 5) Colon surface KD-tree
    colon_surface_mask = get_colon_surface_mask(colon_array_RAI.astype(bool))
    surface_tree, surface_indices = build_colon_surface_kdtree(colon_surface_mask)

    final_labels = np.unique(fluid_cc_clean_array)
    final_labels = final_labels[final_labels != 0]

    new_fluid_array_holes = np.zeros_like(new_fluid_array, dtype=np.uint8)
    new_fluid_array_smooth = np.zeros_like(new_fluid_array, dtype=np.uint8)
    fluid_attached_to_colon = np.zeros_like(new_fluid_array, dtype=np.uint8)

    for lbl in final_labels:
        comp_mask = (fluid_cc_clean_array == lbl)
        comp_indices = np.argwhere(comp_mask)

        comp_distances = colon_distance_array[comp_mask]
        comp_distances = comp_distances[comp_distances > 0]
        if comp_distances.size == 0:
            continue
        min_distance = comp_distances.min()
        min_idx = np.where(comp_distances == min_distance)[0]
        fluid_voxels = comp_indices[min_idx]  # (K, 3) (Z,Y,X) in RAI array ordering

        centroid = fluid_voxels.mean(axis=0)
        central_voxel_index = np.linalg.norm(fluid_voxels - centroid, axis=1).argmin()
        fluid_voxel = fluid_voxels[central_voxel_index]  # (Z,Y,X)

        # Nearest colon surface voxel
        dist, nearest_idx = surface_tree.query(fluid_voxel)
        colon_voxel = surface_indices[nearest_idx]  # (X,Y,Z)? -> note: our arrays are RAI arranged as (X,Y,Z) above

        # Compare along Y (sagittal)
        fluid_y = fluid_voxel[1]
        colon_y = colon_voxel[1]
        position_str = "behind" if fluid_y < colon_y else ("in front of" if fluid_y > colon_y else "same as")
        print(f"Fluid Component {lbl} is {position_str} the colon in the sagittal plane.")

        # Border/erosion-based trimming relative to colon position
        comp_mask_updated = comp_mask.copy()
        border_fluid_colon_mask = np.zeros_like(comp_mask, dtype=np.uint8)

        eroded_mask = binary_erosion(comp_mask_updated, structure=np.ones((3, 3, 3)))
        component_border = comp_mask & ~eroded_mask
        border_indices = np.argwhere(component_border)

        y_min, y_max = border_indices[:, 1].min(), border_indices[:, 1].max()
        erosion = 0

        if fluid_y > colon_y:
            # scan y increasing
            for y in range(y_min, y_max + 1):
                first_time = 1
                while erosion or first_time:
                    first_time = 0
                    if erosion:
                        erosion = 0
                        eroded_mask = binary_erosion(comp_mask_updated, structure=np.ones((3, 3, 3)))
                        component_border = comp_mask_updated & ~eroded_mask
                        component_border[border_fluid_colon_mask == 1] = 0
                        border_indices = np.argwhere(component_border)

                    filtered = border_indices[border_indices[:, 1] == y]
                    for x, z in filtered[:, [0, 2]]:
                        if comp_mask_updated[x, y - 1, z] == 0:
                            count_ones = np.count_nonzero(colon_array_RAI[x - 2:x + 2, y - 3:y, z - 2:z + 2] == 1)
                            if count_ones == 0:
                                comp_mask_updated[x, y, z] = 0
                                erosion = 1
                            else:
                                border_fluid_colon_mask[x, y, z] = 1
        else:
            # scan y decreasing
            for y in range(y_max, y_min - 1, -1):
                first_time = 1
                while erosion or first_time:
                    first_time = 0
                    if erosion:
                        erosion = 0
                        eroded_mask = binary_erosion(comp_mask_updated, structure=np.ones((3, 3, 3)))
                        component_border = comp_mask_updated & ~eroded_mask
                        component_border[border_fluid_colon_mask == 1] = 0
                        border_indices = np.argwhere(component_border)

                    filtered = border_indices[border_indices[:, 1] == y]
                    for x, z in filtered[:, [0, 2]]:
                        if comp_mask_updated[x, y + 1, z] == 0:
                            count_ones = np.count_nonzero(colon_array_RAI[x - 2:x + 2, y + 1:y + 4, z - 2:z + 2] == 1)
                            if count_ones == 0:
                                comp_mask_updated[x, y, z] = 0
                                erosion = 1
                            else:
                                border_fluid_colon_mask[x, y, z] = 1

        # Fill holes per-slice (x-z), then in 3D
        for y in range(comp_mask_updated.shape[1]):
            slice_xz = comp_mask_updated[:, y, :]
            holes_mask = ~slice_xz
            labels2d, num_features = ndimage.label(holes_mask)
            if num_features > 0:
                sizes = np.bincount(labels2d.ravel())
                largest_label = sizes[1:].argmax() + 1 if sizes.size > 1 else 0
                filled_holes = (labels2d != largest_label)
                slice_xz = slice_xz | filled_holes
            comp_mask_updated[:, y, :] = slice_xz

        holes_mask = ~comp_mask_updated
        labels3d, num_features = ndimage.label(holes_mask)
        sizes = np.bincount(labels3d.ravel())
        largest_label = sizes[1:].argmax() + 1 if sizes.size > 1 else 0
        filled_holes = (labels3d != largest_label)
        comp_mask_updated = comp_mask_updated | filled_holes

        new_fluid_array_holes = np.where(comp_mask_updated == 1, comp_mask_updated, new_fluid_array_holes)

        # Smoothing
        smoothed = gaussian_filter(comp_mask_updated.astype(float), sigma=1.0) > 0.5
        new_fluid_array_smooth = np.where(smoothed == 1, smoothed, new_fluid_array_smooth)

        # Attach fluid to colon along Y direction
        smoothed_attach = smoothed.copy()
        eroded_mask = binary_erosion(smoothed, structure=np.ones((3, 3, 3)))
        component_border = smoothed & ~eroded_mask
        border_indices_updated = np.argwhere(component_border)

        if fluid_y > colon_y:
            for y in range(y_min, y_max + 1):
                filtered = border_indices_updated[border_indices_updated[:, 1] == y]
                for x, z in filtered[:, [0, 2]]:
                    if smoothed[x, y - 1, z] == 0:
                        where_colon = np.where(colon_array_RAI[x, y - 3:y, z] == 1)[0]
                        if where_colon.size > 0:
                            first_colon_y = (y - 3) + where_colon[-1]
                            smoothed_attach[x, first_colon_y:y, z] = 1
        else:
            for y in range(y_max, y_min - 1, -1):
                filtered = border_indices_updated[border_indices_updated[:, 1] == y]
                for x, z in filtered[:, [0, 2]]:
                    if smoothed[x, y + 1, z] == 0:
                        where_colon = np.where(colon_array_RAI[x, y + 1:y + 4, z] == 1)[0]
                        if where_colon.size > 0:
                            first_colon_y = y + 1 + where_colon[0]
                            smoothed_attach[x, y:first_colon_y, z] = 1

        fluid_attached_to_colon = np.where(smoothed_attach == 1, smoothed_attach, fluid_attached_to_colon)

    return new_fluid_array_smooth, fluid_attached_to_colon, metadata


def create_RAI_mesh(labelmap: np.ndarray, metadata, output_filename: Path):
    """Create a 3D mesh (VTK) from a binary labelmap array in RAI, save to file."""
    spacing = metadata["spacing"]
    direction = metadata["direction"]
    origin = metadata["origin"]

    # Marching cubes
    verts, faces, _, _ = measure.marching_cubes(labelmap, level=0.5, spacing=spacing)

    # Rotate to RAI and translate by origin
    direction_matrix = np.array(direction).reshape(3, 3)
    verts = np.dot(verts, direction_matrix.T)
    verts += np.array(origin)

    # VTK polydata
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    for v in verts:
        points.InsertNextPoint(*v)
    polydata.SetPoints(points)

    for face in faces:
        tri = vtk.vtkTriangle()
        for i in range(3):
            tri.GetPointIds().SetId(i, int(face[i]))
        cells.InsertNextCell(tri)
    polydata.SetPolys(cells)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(output_filename))
    writer.SetInputData(polydata)
    writer.Write()
    print(f"Mesh saved to {output_filename}")


def save_labelmap_as_mha(numpy_array: np.ndarray, metadata, output_filename: Path):
    """Save a binary labelmap to .mha with metadata."""
    numpy_array = numpy_array.astype(np.uint8)
    sitk_image = sitk.GetImageFromArray(numpy_array)
    sitk_image.SetSpacing(metadata["spacing"])
    sitk_image.SetOrigin(metadata["origin"])
    sitk_image.SetDirection(metadata["direction"])
    sitk.WriteImage(sitk_image, str(output_filename))
    print(f"LabelMap saved: {output_filename}")


def process_directory(
    input_fluid_mha_dir: Path,
    input_colon_mha_dir: Path,
    output_fluid_mesh_dir: Path,
    output_fluid_mha_dir: Path,
    max_files: int = 3,
):
    """
    Process up to `max_files` fluid .mha files: clean/attach fluid and save
    cleaned masks (.mha) and meshes (.vtk).
    """
    output_fluid_mesh_dir.mkdir(parents=True, exist_ok=True)
    output_fluid_mha_dir.mkdir(parents=True, exist_ok=True)

    file_count = 0
    for fluid_path in input_fluid_mha_dir.rglob("*.mha"):
        if file_count >= max_files:
            print(f"Reached the maximum limit of {max_files} files.")
            return

        print(f"Processing file: {fluid_path}")

        # Match colon file by prefix pattern (.*colon_\\d{3})
        m = re.match(r"(.*colon_\d{3})", fluid_path.name)
        if not m:
            raise ValueError(f"Filename format not as expected: {fluid_path.name}")

        colon_basename = f"{m.group(1)}.mha"
        colon_path = input_colon_mha_dir / colon_basename
        if not colon_path.exists():
            print(f"⚠️  Colon reference not found: {colon_path}")
            continue

        out_mesh_path = output_fluid_mesh_dir / (fluid_path.stem + "_cleaned_attached.vtk")
        out_mask_path = output_fluid_mha_dir / (fluid_path.stem + "_cleaned_attached.mha")

        if out_mask_path.exists():
            print(f"Already exists, skipping: {out_mask_path}")
            file_count += 1
            continue

        fluid_array_smooth, fluid_attached_to_colon, metadata = clean_fluid_segmentation(colon_path, fluid_path)

        if np.max(fluid_array_smooth) > 0:
            fluid_array_smooth = (fluid_array_smooth > 0).astype(np.uint8)
            fluid_attached_to_colon = (fluid_attached_to_colon > 0).astype(np.uint8)

            direction = metadata["direction"]

            # revert to original reference
            fluid_attached_reverted = revert_reorientation(fluid_attached_to_colon, direction)
            # (create_RAI_mesh expects RAI array, so we transpose reverted to (x,y,z))
            fluid_attached_reverted_xyz = np.transpose(fluid_attached_reverted, (2, 1, 0))

            save_labelmap_as_mha(fluid_attached_reverted, metadata, out_mask_path)
            create_RAI_mesh(fluid_attached_reverted_xyz, metadata, out_mesh_path)

        file_count += 1


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print(
            "Usage: python postprocess_fluid_segmentation.py "
            "<input_fluid_mha_dir> <input_colon_mha_dir> "
            "<output_fluid_mesh_dir> <output_fluid_mha_dir> [max_files]"
        )
        sys.exit(1)

    input_fluid_mha_dir = Path(sys.argv[1]).resolve()
    input_colon_mha_dir = Path(sys.argv[2]).resolve()
    output_fluid_mesh_dir = Path(sys.argv[3]).resolve()
    output_fluid_mha_dir = Path(sys.argv[4]).resolve()
    max_files = int(sys.argv[5]) if len(sys.argv) > 5 else 145

    print(f"Fluid MHA dir:   {input_fluid_mha_dir}")
    print(f"Colon MHA dir:   {input_colon_mha_dir}")
    print(f"Mesh out dir:    {output_fluid_mesh_dir}")
    print(f"MHA out dir:     {output_fluid_mha_dir}")
    print(f"Max files:       {max_files}")
    print("Starting post-processing...\n")

    process_directory(
        input_fluid_mha_dir,
        input_colon_mha_dir,
        output_fluid_mesh_dir,
        output_fluid_mha_dir,
        max_files=max_files,
    )

    print("\n✅ Done.")
