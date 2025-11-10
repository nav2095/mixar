#!/usr/bin/env python
# coding: utf-8

# In[1]:


# --- Cell 1: Install Dependencies (Run once) ---
# You can comment this out if you've already run it
get_ipython().system('pip install numpy trimesh open3d matplotlib')

# --- Cell 2: Import Libraries ---
import trimesh
import open3d
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Cell 3: Function Definitions (Run this cell) ---
# This cell now contains functions for Task 1, 2, AND 3

# === Configuration ===
N_BINS = 1024
MAX_BIN_INDEX = N_BINS - 1

# === Task 1 Functions ===

def visualize_mesh(mesh, window_title="Mesh Visualization"):
    """
    Visualizes a trimesh object using Open3D.
    """
    # Create an Open3D TriangleMesh object
    o3d_mesh = open3d.geometry.TriangleMesh()
    o3d_mesh.vertices = open3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = open3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    
    print(f"\nVisualizing mesh: '{window_title}'")
    print("  - A new window has opened.")
    print("  - Press 'q' in that window to close it and continue.")
    open3d.visualization.draw_geometries(
        [o3d_mesh],
        window_name=window_title,
        width=1024,
        height=768
    )

def load_and_inspect_mesh(filepath):
    """
    Performs all steps for Task 1.
    Returns the loaded mesh object on success, else None.
    """
    print(f"--- Task 1: Loading and Inspecting Mesh ---")
    print(f"Loading: {filepath}")

    if not os.path.exists(filepath):
        print(f"\n*** ERROR: File not found. ***")
        return None

    try:
        mesh = trimesh.load_mesh(filepath, force='mesh')
        vertices = mesh.vertices
        
        print(f"\nLoad successful.")
        print("="*40)
        print("  Mesh Statistics (as required by Task 1)")
        print("="*40)
        print(f"  - Number of vertices: {len(vertices)}")
        
        if len(vertices) > 0:
            min_vals = np.min(vertices, axis=0)
            max_vals = np.max(vertices, axis=0)
            mean_vals = np.mean(vertices, axis=0)
            std_vals = np.std(vertices, axis=0)
            
            print(f"  - Min (X, Y, Z):    {min_vals}")
            print(f"  - Max (X, Y, Z):    {max_vals}")
            print(f"  - Mean (X, Y, Z):   {mean_vals}")
            print(f"  - Std Dev (X, Y, Z):{std_vals}")
        else:
            print("  - Mesh contains no vertex data.")
            return None
        
        print("="*40)
        visualize_mesh(mesh, window_title=f"Original Mesh: {os.path.basename(filepath)}")
        print("\n--- Task 1 Complete ---")
        return mesh

    except Exception as e:
        print(f"\n*** An error occurred during processing: ***\n{e}")
        return None

# === Task 2 Functions ===

def save_mesh(mesh, filepath):
    """
    Saves a trimesh object to a file.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        mesh.export(filepath)
        print(f"  - Successfully saved: {filepath}")
    except Exception as e:
        print(f"  - *** ERROR saving {filepath}: {e} ***")

def visualize_quantized_mesh_effect(quantized_vertices, faces, normalization_type, window_title="Quantized Mesh Effect"):
    """
    Visualizes the *effect* of quantization by mapping integer bins
    back to their normalized float range [0,1] or [-1,1].
    """
    display_vertices_0_1 = (quantized_vertices.astype(np.float64) + 0.5) / N_BINS

    if normalization_type == 'min_max':
        display_vertices = display_vertices_0_1
    elif normalization_type == 'unit_sphere':
        display_vertices = (display_vertices_0_1 * 2.0) - 1.0
    else:
        raise ValueError("Invalid normalization_type.")
        
    quantized_effect_mesh = trimesh.Trimesh(vertices=display_vertices, faces=faces)
    visualize_mesh(quantized_effect_mesh, window_title=window_title)

def min_max_normalize(vertices):
    """
    Applies Min-Max normalization to vertices, scaling them to the [0, 1] range.
    """
    min_vals = vertices.min(axis=0)
    max_vals = vertices.max(axis=0)
    data_range = max_vals - min_vals
    # Avoid division by zero for flat meshes
    data_range[data_range == 0] = 1e-8
    normalized_vertices = (vertices - min_vals) / data_range
    return normalized_vertices, min_vals, data_range

def unit_sphere_normalize(vertices):
    """
    Applies Unit Sphere normalization, scaling to a [-1, 1] sphere.
    """
    centroid = vertices.mean(axis=0)
    centered_vertices = vertices - centroid
    max_dist = np.max(np.linalg.norm(centered_vertices, axis=1))
    if max_dist == 0:
        max_dist = 1e-8
    normalized_vertices = centered_vertices / max_dist
    return normalized_vertices, centroid, max_dist

def quantize(vertices, normalization_type):
    """
    Quantizes normalized vertices into integer bins (0 to 1023).
    """
    if normalization_type == 'min_max':
        # [0, 1] -> [0, 1023]
        scaled_vertices = vertices * MAX_BIN_INDEX
    elif normalization_type == 'unit_sphere':
        # [-1, 1] -> [0, 1] -> [0, 1023]
        scaled_vertices = (vertices + 1.0) / 2.0
        scaled_vertices = scaled_vertices * MAX_BIN_INDEX
    else:
        raise ValueError("Invalid normalization_type.")
        
    quantized = np.floor(scaled_vertices)
    quantized = np.clip(quantized, 0, MAX_BIN_INDEX)
    return quantized.astype(np.uint16)

# === Task 3 Functions ===

def dequantize(quantized_vertices, normalization_type):
    """
    Dequantizes integer bins back to the normalized float range.
    Maps to the center of the bin for higher accuracy.
    """
    # First, convert [0, 1023] int range to [0, 1] float range
    # (e.g., 0 -> 0.000488, 1023 -> 0.999511)
    dequantized_0_1 = (quantized_vertices.astype(np.float64) + 0.5) / N_BINS
    
    if normalization_type == 'min_max':
        # Desired range is [0, 1].
        return dequantized_0_1
    elif normalization_type == 'unit_sphere':
        # Desired range is [-1, 1]. Map [0, 1] -> [-1, 1]
        return (dequantized_0_1 * 2.0) - 1.0
    else:
        raise ValueError("Invalid normalization_type.")

def denormalize_min_max(vertices, min_vals, data_range):
    """
    Reverses Min-Max normalization.
    """
    return (vertices * data_range) + min_vals

def denormalize_unit_sphere(vertices, centroid, max_dist):
    """
    Reverses Unit Sphere normalization.
    """
    return (vertices * max_dist) + centroid

def compute_error(original_vertices, reconstructed_vertices):
    """
    Computes Mean Squared Error (MSE) between original and reconstructed vertices.
    """
    squared_error = (original_vertices - reconstructed_vertices) ** 2
    total_mse = np.mean(squared_error)
    per_axis_mse = np.mean(squared_error, axis=0)
    return total_mse, per_axis_mse

def visualize_comparison(original_mesh, reconstructed_mesh, title="Mesh Comparison"):
    """
    Displays the original and reconstructed meshes side-by-side as point clouds.
    Original: Red, Reconstructed: Blue
    """
    # 1. Create Open3D PointCloud objects
    o3d_original = open3d.geometry.PointCloud()
    o3d_original.points = open3d.utility.Vector3dVector(original_mesh.vertices)
    o3d_original.paint_uniform_color([1, 0, 0]) # Red
    
    o3d_reconstructed = open3d.geometry.PointCloud()
    o3d_reconstructed.points = open3d.utility.Vector3dVector(reconstructed_mesh.vertices)
    o3d_reconstructed.paint_uniform_color([0, 0, 1]) # Blue
    
    # 2. Translate reconstructed mesh for side-by-side view
    bbox = o3d_original.get_axis_aligned_bounding_box()
    translation = np.array([bbox.get_extent()[0] * 1.2, 0, 0])
    o3d_reconstructed.translate(translation)
    
    # 3. Visualize
    print(f"\nVisualizing: '{title}'")
    print("  - Original (Red) vs. Reconstructed (Blue)")
    print("  - A new window has opened.")
    print("  - Press 'q' in that window to close it and continue.")
    open3d.visualization.draw_geometries(
        [o3d_original, o3d_reconstructed],
        window_name=title,
        width=1280,
        height=720
    )

def plot_error_per_axis(mse_mm, mse_us, filename):
    """
    Generates a bar plot comparing per-axis MSE for both methods.
    """
    labels = ['X-axis', 'Y-axis', 'Z-axis']
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, mse_mm, width, label='Min-Max')
    rects2 = ax.bar(x + width/2, mse_us, width, label='Unit Sphere')
    
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title(f'Per-Axis Reconstruction Error ({filename})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Use a log scale for Y-axis to see differences better
    ax.set_yscale('log')
    ax.set_ylabel('Mean Squared Error (MSE) - Log Scale')
    
    fig.tight_layout()
    
    # Save the plot to the output directory
    output_path = os.path.join("task_3_output", f"{os.path.basename(filename).split('.')[0]}_error_plot.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"\nError plot saved to: {output_path}")
    
    plt.show()


# --- Cell 4: Execute Task 1 (Run this cell) ---
#
# *** SET YOUR FILE PATH HERE ***
# I've used the 'r' prefix to handle the Windows backslashes.
filepath = r"C:\Users\aadim\Downloads\8samples\8samples\girl.obj"

# --- Run the analysis ---
# This will print all stats and open the visualization window.
# We store the mesh in a variable to use in Task 2.
original_mesh = load_and_inspect_mesh(filepath)


# --- Cell 5: Execute Task 2 (Run this cell after Task 1) ---
#
# This cell now saves the normalization parameters for Task 3
#

if original_mesh:
    print(f"\n--- Task 2: Normalizing and Quantizing Mesh ---")
    
    base_filename = os.path.basename(filepath).split('.')[0]
    output_dir = "task2_output"
    
    original_vertices = original_mesh.vertices
    original_faces = original_mesh.faces

    # --- 1. Min-Max (Capture parameters) ---
    print("\nProcessing Min-Max Pipeline...")
    # ***MODIFIED:*** Storing min_vals and data_range
    norm_v_mm, min_vals_mm, data_range_mm = min_max_normalize(original_vertices)
    quant_v_mm = quantize(norm_v_mm, 'min_max')
    
    norm_mesh_mm = trimesh.Trimesh(vertices=norm_v_mm, faces=original_faces)
    quant_mesh_mm = trimesh.Trimesh(vertices=quant_v_mm, faces=original_faces)
    
    print("Saving Min-Max deliverables...")
    save_mesh(norm_mesh_mm, os.path.join(output_dir, f"{base_filename}_normalized_minmax.obj"))
    save_mesh(quant_mesh_mm, os.path.join(output_dir, f"{base_filename}_quantized_minmax.obj"))
    
    visualize_mesh(norm_mesh_mm, "Normalized Mesh (Min-Max, [0, 1] space)")
    visualize_quantized_mesh_effect(quant_v_mm, original_faces, 'min_max', 
                                    "Quantization Effect (Min-Max)")

    # --- 2. Unit Sphere (Capture parameters) ---
    print("\nProcessing Unit Sphere Pipeline...")
    # ***MODIFIED:*** Storing centroid and max_dist
    norm_v_us, centroid_us, max_dist_us = unit_sphere_normalize(original_vertices)
    quant_v_us = quantize(norm_v_us, 'unit_sphere')

    norm_mesh_us = trimesh.Trimesh(vertices=norm_v_us, faces=original_faces)
    quant_mesh_us = trimesh.Trimesh(vertices=quant_v_us, faces=original_faces)
    
    print("Saving Unit Sphere deliverables...")
    save_mesh(norm_mesh_us, os.path.join(output_dir, f"{base_filename}_normalized_unitsphere.obj"))
    save_mesh(quant_mesh_us, os.path.join(output_dir, f"{base_filename}_quantized_unitsphere.obj"))

    visualize_mesh(norm_mesh_us, "Normalized Mesh (Unit Sphere, [-1, 1] space)")
    visualize_quantized_mesh_effect(quant_v_us, original_faces, 'unit_sphere', 
                                    "Quantization Effect (Unit Sphere)")
    
    print("\n--- Task 2 Complete ---")

else:
    print("\n--- Task 2 Skipped ---")
    print("Run Cell 4 successfully to load a mesh before running Task 2.")


# --- Cell 6: Execute Task 3 (Run this cell after Task 2) ---
#
# This cell performs all steps for Task 3:
# 1. Dequantize & Denormalize
# 2. Compute MSE
# 3. Visualize reconstructed meshes
# 4. Plot reconstruction error
#

# Check that Task 1 & 2 have been run
if 'original_mesh' in locals() and 'quant_v_mm' in locals():
    print(f"\n--- Task 3: Dequantize, Denormalize, and Measure Error ---")
    
    # --- 1. Min-Max Pipeline ---
    print("\nReconstructing from Min-Max...")
    dequant_v_mm = dequantize(quant_v_mm, 'min_max')
    recon_v_mm = denormalize_min_max(dequant_v_mm, min_vals_mm, data_range_mm)
    
    # --- 2. Unit Sphere Pipeline ---
    print("Reconstructing from Unit Sphere...")
    dequant_v_us = dequantize(quant_v_us, 'unit_sphere')
    recon_v_us = denormalize_unit_sphere(dequant_v_us, centroid_us, max_dist_us)
    
    # --- 3. Compute Error (MSE) ---
    print("\nCalculating Reconstruction Error...")
    original_vertices = original_mesh.vertices # ensure we have the original
    mse_mm, axis_mse_mm = compute_error(original_vertices, recon_v_mm)
    mse_us, axis_mse_us = compute_error(original_vertices, recon_v_us)
    
    print("\n" + "="*50)
    print("  Reconstruction Error Report (MSE)")
    print("="*50)
    print(f"  [Min-Max] Total MSE:     {mse_mm: .10e}")
    print(f"  [Min-Max] Per-Axis MSE:  {axis_mse_mm}")
    print(f"  [Unit Sphere] Total MSE: {mse_us: .10e}")
    print(f"  [Unit Sphere] Per-Axis MSE: {axis_mse_us}")
    print("="*50)
    
    # --- 4. Plot Reconstruction Error ---
    print("\nGenerating error plot...")
    plot_error_per_axis(axis_mse_mm, axis_mse_us, filepath)
    
    # --- 5. Visualize Reconstructed Meshes ---
    # Create trimesh objects for visualization
    recon_mesh_mm = trimesh.Trimesh(vertices=recon_v_mm, faces=original_faces)
    recon_mesh_us = trimesh.Trimesh(vertices=recon_v_us, faces=original_faces)
    
    print("\nLaunching final comparison visualizations...")
    visualize_comparison(original_mesh, recon_mesh_mm, 
                         "Min-Max Reconstruction (Original-Red, Recon-Blue)")
    
    visualize_comparison(original_mesh, recon_mesh_us, 
                         "Unit Sphere Reconstruction (Original-Red, Recon-Blue)")

    print("\n--- Task 3 Complete ---")
    
else:
    print("\n--- Task 3 Skipped ---")
    print("Run Cell 4 and Cell 5 successfully before running Task 3.")


# In[ ]:




