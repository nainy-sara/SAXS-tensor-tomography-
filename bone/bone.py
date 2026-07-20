# =====================================================
# BEFORE RUNNING THIS SCRIPT:
# 1. Install required packages:
#    pip install mumott h5py matplotlib numpy ipywidgets colorcet
# 2. Download the dataset manually from:
#    https://zenodo.org/records/10074598/files/trabecular_bone_9.h5
#    and place it in the same folder as this script.
# =====================================================

import h5py
import matplotlib.pyplot as plt
import numpy as np
import colorcet
from mumott.data_handling import DataContainer
from mumott.methods.basis_sets import SphericalHarmonics
from mumott.methods.projectors import SAXSProjector
from mumott.methods.residual_calculators import GradientResidualCalculator
from mumott.optimization.loss_functions import SquaredLoss
from mumott.optimization.optimizers import LBFGS
from mumott.optimization.regularizers import Laplacian


# 1. Load the dataset (make sure the file is in the current directory)
data_container = DataContainer('trabecular_bone_9.h5')

# 2. Initial setup and optimization (150 iterations)
basis_set = SphericalHarmonics(ell_max=6)
projector = SAXSProjector(data_container.geometry)
residual_calculator = GradientResidualCalculator(
    data_container=data_container,
    basis_set=basis_set,
    projector=projector
)
loss_function = SquaredLoss(residual_calculator)
regularizer = Laplacian()
loss_function.add_regularizer(
    name='laplacian',
    regularizer=regularizer,
    regularization_weight=5e-2
)

optimizer = LBFGS(loss_function, maxiter=150)
results = optimizer.optimize()

# 3. Extract outputs
output = basis_set.get_output(residual_calculator.coefficients)
scale = output.mean_intensity
color = output.fractional_anisotropy
orientation = output.eigenvector_3

# 4. Plot images (3 different slices)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, idx in enumerate([20, 25, 30]):
    im1 = axes[0, i].imshow(scale[:, idx, :], cmap='cet_gouldian', origin='lower')
    axes[0, i].set_title(f'Intensity (Slice {idx})')
    axes[0, i].axis('off')
    plt.colorbar(im1, ax=axes[0, i], fraction=0.046)
    
    im2 = axes[1, i].imshow(color[:, idx, :], cmap='cet_fire', origin='lower', vmin=0, vmax=0.6)
    axes[1, i].set_title(f'Fractional Anisotropy (Slice {idx})')
    axes[1, i].axis('off')
    plt.colorbar(im2, ax=axes[1, i], fraction=0.046)

plt.suptitle('SAXS Tensor Tomography Reconstruction - Trabecular Bone', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('saxstt_results.png', dpi=300)
plt.show()

# 5. Plot the orientation vector field (Quiver Plot)
slice_idx = 25
mask = scale[:, slice_idx, :] > 0.3
x, y = np.meshgrid(np.arange(scale.shape[2]), np.arange(scale.shape[0]), indexing='ij')
ux = orientation[:, slice_idx, :, 0] * mask
uy = orientation[:, slice_idx, :, 2] * mask

fig2, ax2 = plt.subplots(1, figsize=(10, 9))
q = ax2.quiver(x[::3, ::3], y[::3, ::3], 
               ux[::3, ::3], uy[::3, ::3],
               color[::3, slice_idx, ::3], 
               cmap='cet_rainbow4', scale=20, width=0.004,
               pivot='mid', clim=(0, 0.6))
ax2.set_title('Orientation Field (Color = Fractional Anisotropy)', fontsize=14)
ax2.axis('off')
plt.colorbar(q, ax=ax2, fraction=0.03)
plt.savefig('orientation_quiver.png', dpi=300)
plt.show()

print("Done! Images saved as 'saxstt_results.png' and 'orientation_quiver.png'.")
