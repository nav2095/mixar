# 3D Mesh Normalization and Quantization Analysis

This project implements a complete end-to-end pipeline for 3D mesh processing, including:

- Loading and visualizing 3D mesh models
- Applying two normalization methods:
  - **Min-Max Normalization**
  - **Unit Sphere Normalization**
- Quantization and Dequantization of mesh vertex data
- Reconstruction of the mesh from quantized data
- Measurement of geometric distortion using **Mean Squared Error (MSE)**
- Visual comparison of original vs reconstructed meshes

Both step-by-step and automated execution workflows are included.

---

## 📂 Project Structure

├── task_scripts.py # Step-by-step notebook workflow
├── final stanalone script.py # Fully automated pipeline script
├── README.md # This file
├── task2_output/ # Normalized & quantized output models
│ ├── model_normalized_minmax.obj
│ ├── model_quantized_minmax.obj
│ ├── model_normalized_unitsphere.obj
│ ├── model_quantized_unitsphere.obj
│ └── ...
│
├── task_3_output/ # Reconstruction error plots
│ ├── model_error_plot.png
│ └── ...
│
├── screenshots/ # Visualization screenshots
│ ├── original_mesh.png
│ ├── reconstructed_comparison.png
│ └── ...
│
└── *.obj # Input mesh model files

yaml
Copy code

---

## 🔧 Installation

Install required Python libraries:

```bash
pip install numpy trimesh open3d matplotlib
▶️ How to Run
A. Step-by-Step Notebook Use
Open:

Copy code
mesh_analysis_notebook.py
Set the .obj file path inside the notebook.

Run each cell to observe:

Normalization

Quantization

Reconstruction quality

MSE values

Visual comparison

B. Automatic Full Pipeline Execution
If the .obj file is in the same directory:

bash
Copy code
python mesh_processing_pipeline.py model.obj
If located elsewhere:

bash
Copy code
python mesh_processing_pipeline.py "C:\path\to\your\model.obj"
The script will:

Normalize (Min-Max and Unit Sphere)

Quantize and Reconstruct

Print MSE values

Save output files

Display meshes
(Press q to close each visualization window)

🧠 Key Findings
Method	Reconstruction Quality	Notes
Min-Max Normalization	Lower	Sensitive to aspect ratio; loses detail on smaller axes.
Unit Sphere Normalization	Higher (Recommended) ✅	Preserves geometry uniformly, greatly reducing quantization error.

Conclusion
Unit Sphere normalization consistently produces better, more accurate reconstructions than Min-Max normalization because it scales the mesh uniformly in all directions before quantization.

📎 Output Provided
Normalized Mesh Files (.obj)

Quantized Mesh Files (.obj)

Reconstructed Mesh Files (.obj)

Error Plots (.png)

Visual Comparison Screenshots

