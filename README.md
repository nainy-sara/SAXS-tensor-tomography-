# SAXS-tensor-tomography-

# README: SAXS Tensor Tomography Analysis with MUMOTT

## Overview

This notebook demonstrates **SAXS (Small-Angle X-ray Scattering) tensor tomography** using the **MUMOTT** (Multi-Modal Operator Tomographic Tool) library. It performs 3D reconstruction and analysis of tensor properties from X-ray scattering data, visualizing mean intensity, fractional anisotropy, and molecular orientation.

## Prerequisites

### Required Libraries
- `mumott` - Main tomographic reconstruction library
- `h5py` - HDF5 file handling
- `matplotlib` - Data visualization
- `numpy` - Numerical computing
- `ipywidgets` - Interactive widgets
- `colorcet` - Perceptually uniform colormaps

Install all dependencies with:
```bash
pip install mumott h5py matplotlib numpy ipywidgets colorcet
```

## Workflow

### 1. **Data Loading**
   - Downloads the SAXS dataset (`saxstt_dataset_M.h5`) from Zenodo
   - Creates a `DataContainer` object containing 417 projections
   - Visualizes raw projections using `ProjectionViewer`

### 2. **Reconstruction Setup**
   - **Basis Set**: Spherical harmonics (ℓ_max = 6) decompose the scattering signal
   - **Projector**: `SAXSProjector` models the X-ray scattering geometry
   - **Residual Calculator**: `GradientResidualCalculator` computes misfit between model and data
   - **Loss Function**: `SquaredLoss` with Laplacian regularization (weight = 0.1)

### 3. **Optimization**
   - Uses **L-BFGS** optimizer with 10 iterations
   - Minimizes loss function to recover 3D coefficient field
   - Converges after ~12 minutes on GPU

### 4. **Reconstruction Outputs**
   The `basis_set.get_output()` method extracts physical quantities:
   - **`eigenvector_3`**: Principal orientation tensor direction
   - **`mean_intensity`**: Average scattering intensity (scale)
   - **`fractional_anisotropy`**: Degree of preferred orientation (color)

### 5. **Visualization**
   - **Slice Maps**: Intensity and anisotropy along a cross-section (y = 25)
   - **Quiver Plot**: Vector field showing molecular orientation with magnitude and color encoding anisotropy

## Key Output Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `mean_intensity` | Average X-ray scattering strength | 0.3 - 1.0 (clipped) |
| `fractional_anisotropy` | Preferred orientation strength | 0.0 - 0.6 (clipped) |
| `eigenvector_3` | Primary molecular orientation | Unit vector (x, y, z) |

## File Structure

```
saxstt_dataset_M.h5       # Input: Experimental SAXS data (180 MB)
mumott.ipynb              # Main analysis notebook
```

## References

- **Dataset**: [Zenodo Record 7326784](https://zenodo.org/records/7326784)
- **MUMOTT Documentation**: [Official MUMOTT Repository](https://github.com/3dct/mumott)

## Deprecation Warnings

The notebook may display warnings for deprecated MUMOTT and matplotlib features:
- Replace `rotations` → `inner_angle`
- Replace `tilts` → `outer_angle`
- Replace `offset_j` → `j_offset`
- Replace `offset_k` → `k_offset`
- Use `matplotlib.colormaps()` instead of deprecated `get_cmap()`

## Performance Notes

- Total runtime: ~12-15 minutes (depends on hardware)
- Optimization: 10 iterations × ~70 seconds per iteration
- GPU acceleration recommended for production use

## Customization

Adjust reconstruction parameters:
```python
basis_set = SphericalHarmonics(ell_max=8)  # Higher order harmonics
regularizer_weight = 1e-2                   # Stronger/weaker smoothing
optimizer = LBFGS(loss_function, maxiter=20) # More iterations
```
