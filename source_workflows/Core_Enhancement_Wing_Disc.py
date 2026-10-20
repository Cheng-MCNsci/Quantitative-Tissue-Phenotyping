# Exported notebook workflow: Core_Enhancement_Wing_Disc.ipynb

# %% Cell 2
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt

plane_files = {
    "top": Path("top_Z28-Z29_4channels.tif"),
    "middle": Path("middle_Z33-Z34_4channels.tif"),
    "bottom": Path("bottom_Z38-Z39_4channels.tif"),
}
channel_names = ["GFP", "unknown", "DAPI", "RFP"]
dapi_channel_index = 2  # C3, zero-based
pixel_size_um = 0.5546006944444444


# %% Cell 4
raw_planes = {}

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

for ax, (name, path) in zip(axes, plane_files.items()):
    with tifffile.TiffFile(path) as tif:
        image = tif.asarray()
        print(f"{name}: shape={image.shape}, dtype={image.dtype}, "
              f"axes={tif.series[0].axes}")

    assert image.shape == (4, 1152, 2189), f"Unexpected shape: {path}"
    assert image.dtype == np.uint16, f"Unexpected dtype: {path}"
    raw_planes[name] = image

    dapi = image[dapi_channel_index]
    vmin, vmax = np.percentile(dapi, [1, 99.8])

    ax.imshow(dapi, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"{name.capitalize()} — raw DAPI")
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% Cell 5
# Raw DAPI crop inspection only; no enhancement.
from matplotlib.patches import Rectangle
crop_size_px = 192
crop_centers_xy = [(650, 800), (1100, 600), (1500, 500)]  # editable (x, y)
dapi_inspection_crops = {}
for name, image in raw_planes.items():
    dapi = image[dapi_channel_index]
    lo, hi = np.percentile(dapi, [1, 99.8])
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(dapi, cmap='gray', vmin=lo, vmax=hi)
    axes[0].set_title(f'{name.capitalize()} — crop locations')
    dapi_inspection_crops[name] = []
    for i, (cx, cy) in enumerate(crop_centers_xy):
        x0, y0 = cx - 96, cy - 96
        crop = dapi[y0:y0+192, x0:x0+192]
        dapi_inspection_crops[name].append({'bounds_xy': (x0, y0, x0+192, y0+192), 'raw': crop.copy()})
        axes[0].add_patch(Rectangle((x0, y0), 192, 192, fill=False, edgecolor='cyan'))
        axes[0].text(x0, y0-10, str(i+1), color='cyan')
        axes[i+1].imshow(crop, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
        axes[i+1].set_title(f'Crop {i+1}: x={x0}, y={y0}')
        axes[i+1].plot([10, 10+10/pixel_size_um], [177,177], color='yellow', linewidth=3)
        axes[i+1].text(10,170,'10 um',color='yellow')
    for ax in axes:
        ax.axis('off')
    fig.suptitle(f'{name.capitalize()} raw DAPI — same display contrast within plane')
    plt.tight_layout()
    plt.show()
print('Raw pixels unchanged. Check the boxed locations before treating crops as representative.')

# %% Cell 6
# Exploratory enhancement comparison on ONE crop; no segmentation or export.
from scipy.ndimage import gaussian_filter, gaussian_laplace

preview_plane = 'top'
preview_crop_index = 0
crop = dapi_inspection_crops[preview_plane][preview_crop_index]['raw'].astype(np.float32)

# Trial scales in pixels; these are not finalized nuclear-size settings.
smooth_sigma_px = 0.7
background_sigma_px = 8.0
core_sigma_px = 1.5
smoothed = gaussian_filter(crop, smooth_sigma_px)
background = gaussian_filter(smoothed, background_sigma_px)
background_removed = np.maximum(smoothed - background, 0)
# Positive Laplacian-of-Gaussian response emphasizes bright blob centers.
# It is a seed/core response, not a nuclear boundary image.
core_response = np.maximum(-gaussian_laplace(background_removed, core_sigma_px) * core_sigma_px**2, 0)

preview_images = [crop, smoothed, background_removed, core_response]
preview_titles = ['Raw DAPI', 'Gentle Gaussian smoothing', 'Smoothed + background subtraction', 'LoG core response (not a mask)']
raw_limits = np.percentile(crop, [1, 99.8])
fig, axes = plt.subplots(1, 4, figsize=(16, 5), dpi=120)
for i, (ax, img, title) in enumerate(zip(axes, preview_images, preview_titles)):
    limits = raw_limits if i < 2 else (0, np.percentile(img, 99.8))
    ax.imshow(img, cmap='gray', vmin=limits[0], vmax=max(limits[1], limits[0]+1e-6), interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.plot([10, 10+10/pixel_size_um], [177,177], color='yellow', linewidth=3)
    ax.text(10,170,'10 um',color='yellow')
    ax.axis('off')
fig.suptitle('Top / crop 1 — exploratory only; processed panels use their own display range')
plt.tight_layout()
plt.show()
print(f'Trial scales: smoothing {smooth_sigma_px*pixel_size_um:.2f} um; background {background_sigma_px*pixel_size_um:.2f} um; core {core_sigma_px*pixel_size_um:.2f} um (Gaussian sigmas).')
print('Check for one core per nucleus, preserved weak nuclei, and no merging. Crop-edge pixels are not suitable for evaluation.')
print('No raw channels modified; no masks or files created.')

# %% Cell 7
# Fine-tuning preview only. Filter the full plane, then inspect a smaller crop.
# No peak detection, segmentation, or export.
tune_plane = 'top'
bounds = dapi_inspection_crops[tune_plane][0]['bounds_xy']
x0, y0, x1, y1 = bounds
# Central 96 x 96 pixels of the previous crop (~53 um across).
xs = slice(x0 + 48, x1 - 48)
ys = slice(y0 + 48, y1 - 48)
full_dapi = raw_planes[tune_plane][dapi_channel_index].astype(np.float32)
raw_tune_crop = full_dapi[ys, xs]
smoothing_trials_px = [0.5, 1.0]
core_trials_px = [1.0, 1.5, 2.0, 2.5]
background_trial_px = 12.0  # broad background estimate; held fixed for this sweep
core_tuning_results = {}


# Raw reference, enlarged and unfiltered.
fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
lo, hi = np.percentile(raw_tune_crop, [1, 99.8])
ax.imshow(raw_tune_crop, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
ax.set_title(f'{tune_plane} raw DAPI reference — 96 x 96 px')
ax.axis('off')
plt.show()


fig, axes = plt.subplots(2, 4, figsize=(16, 9), dpi=150)
for row, smooth_sigma in enumerate(smoothing_trials_px):
    smooth_full = gaussian_filter(full_dapi, smooth_sigma)
    corrected_full = np.maximum(smooth_full - gaussian_filter(smooth_full, background_trial_px), 0)
    row_responses = []
    for core_sigma in core_trials_px:
        response = np.maximum(-gaussian_laplace(corrected_full, core_sigma) * core_sigma**2, 0)[ys, xs]
        core_tuning_results[(smooth_sigma, core_sigma)] = response.copy()
        row_responses.append(response)
    # Same intensity limits for every response in a row; no per-panel stretching.
    shared_vmax = np.percentile(np.stack(row_responses), 99.5)
    for col, (core_sigma, response) in enumerate(zip(core_trials_px, row_responses)):
        ax = axes[row, col]
        ax.imshow(response, cmap='gray', vmin=0, vmax=max(shared_vmax, 1e-6), interpolation='nearest')
        ax.set_title(f'Smooth sigma {smooth_sigma} px; core sigma {core_sigma} px; core scale {core_sigma*pixel_size_um:.2f} um')
        ax.plot([5, 5+5/pixel_size_um], [88,88], color='yellow', linewidth=3)
        ax.text(5,82,'5 um',color='yellow')
        ax.axis('off')
fig.suptitle('Core-scale sweep: compare each bright core against the raw reference; settings remain exploratory')
plt.tight_layout()
plt.show()
print('Inspect: multiple cores within one nucleus at small scales; merged neighboring nuclei or missing weak nuclei at large scales.')
print('Gaussian sigma is a filter scale, not nuclear diameter. No masks created; raw pixels unchanged.')

# %% Cell 8
# Select only middle-plane crop #2 for subsequent tuning.
selected_plane = 'middle'
selected_crop_number = 2
selected_crop_info = dapi_inspection_crops[selected_plane][selected_crop_number - 1]
selected_bounds_xy = selected_crop_info['bounds_xy']
selected_raw_dapi = selected_crop_info['raw'].copy()
x0, y0, x1, y1 = selected_bounds_xy
print(f'Selected: middle crop #2; x={x0}:{x1}, y={y0}:{y1}; shape={selected_raw_dapi.shape}')
fig, ax = plt.subplots(figsize=(7, 7), dpi=130)
lo, hi = np.percentile(selected_raw_dapi, [1, 99.8])
ax.imshow(selected_raw_dapi, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
ax.set_title('Middle plane — crop #2 — raw DAPI')
h = selected_raw_dapi.shape[0]
ax.plot([10, 10 + 10 / pixel_size_um], [h - 15, h - 15], color='yellow', linewidth=3)
ax.text(10, h - 22, '10 um', color='yellow')
ax.axis('off')
plt.tight_layout()
plt.show()
# Selection and display only. No enhancement or segmentation.

# %% Cell 9
# Middle crop #2: raw (left) and enhanced core response (right).
# Change ONE setting at a time, then rerun this cell.
smoothing_sigma_px = 0.5
background_sigma_px = 12
core_sigma_px = 1.8

from scipy.ndimage import gaussian_filter, gaussian_laplace

# Filter the full plane before cropping to avoid crop-edge artifacts.
source_dapi = raw_planes[selected_plane][dapi_channel_index].astype(np.float32)
smoothed_dapi = gaussian_filter(source_dapi, smoothing_sigma_px)
background_dapi = gaussian_filter(smoothed_dapi, background_sigma_px)
corrected_dapi = np.maximum(smoothed_dapi - background_dapi, 0)
core_response_full = np.maximum(
    -gaussian_laplace(corrected_dapi, core_sigma_px) * core_sigma_px**2, 0
)
x0, y0, x1, y1 = selected_bounds_xy
raw_comparison = source_dapi[y0:y1, x0:x1]
enhanced_comparison = core_response_full[y0:y1, x0:x1]

fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=130)
raw_lo, raw_hi = np.percentile(raw_comparison, [1, 99.8])
enhanced_hi = max(float(np.percentile(enhanced_comparison, 99.8)), 1e-6)
axes[0].imshow(raw_comparison, cmap='gray', vmin=raw_lo, vmax=raw_hi, interpolation='nearest')
axes[1].imshow(enhanced_comparison, cmap='gray', vmin=0, vmax=enhanced_hi, interpolation='nearest')
axes[0].set_title('Raw DAPI — middle crop #2')
axes[1].set_title('Enhanced core response — middle crop #2')
for ax in axes:
    h = raw_comparison.shape[0]
    ax.plot([10, 10 + 10 / pixel_size_um], [h - 15, h - 15], color='yellow', linewidth=3)
    ax.text(10, h - 22, '10 um', color='yellow')
    ax.axis('off')
fig.suptitle(f'Smoothing sigma={smoothing_sigma_px} px; background sigma={background_sigma_px} px; core sigma={core_sigma_px} px')
plt.tight_layout()
plt.show()
print('Each panel uses its own display range: compare spatial structure, not brightness.')
print('Enhanced image is a core response, not a nuclear mask. Raw channels unchanged; no files exported.')


# %% Cell 10
# Entire middle plane: raw DAPI vs enhanced core response.
# Agreed settings; display only, no segmentation or export.
smoothing_sigma_px = 0.5
background_sigma_px = 12.0
core_sigma_px = 1.8

from scipy.ndimage import gaussian_filter, gaussian_laplace
middle_raw_dapi = raw_planes['middle'][dapi_channel_index].copy()
middle_smoothed_dapi = gaussian_filter(middle_raw_dapi.astype(np.float32), smoothing_sigma_px)
middle_background_dapi = gaussian_filter(middle_smoothed_dapi, background_sigma_px)
middle_corrected_dapi = np.maximum(middle_smoothed_dapi - middle_background_dapi, 0)
middle_enhanced_dapi = np.maximum(
    -gaussian_laplace(middle_corrected_dapi, core_sigma_px) * core_sigma_px**2, 0
)

fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=150)
raw_lo, raw_hi = np.percentile(middle_raw_dapi, [1, 99.8])
enhanced_hi = max(float(np.percentile(middle_enhanced_dapi, 99.8)), 1e-6)
axes[0].imshow(middle_raw_dapi, cmap='gray', vmin=raw_lo, vmax=raw_hi, interpolation='nearest')
axes[1].imshow(middle_enhanced_dapi, cmap='gray', vmin=0, vmax=enhanced_hi, interpolation='nearest')
axes[0].set_title('Middle Z33-Z34 — raw DAPI')
axes[1].set_title('Middle Z33-Z34 — enhanced core response')
for ax in axes:
    h = middle_raw_dapi.shape[0]
    ax.plot([80, 80 + 100 / pixel_size_um], [h - 80, h - 80], color='yellow', linewidth=3)
    ax.text(80, h - 105, '100 um', color='yellow')
    ax.axis('off')
fig.suptitle('Smoothing sigma=0.5 px; background sigma=12 px; core sigma=1.8 px')
plt.tight_layout()
plt.show()
print(f'Both arrays: {middle_raw_dapi.shape}; enhanced dtype: {middle_enhanced_dapi.dtype}')
print('Panels use separate display ranges. Compare structure, not fluorescence intensity.')
print('Core response is not a nuclear mask. Raw channels unchanged; no files exported.')


# %% Cell 11
# Apply approved enhancement to all three planes and export DAPI only.
import json
from scipy.ndimage import gaussian_filter, gaussian_laplace

settings = dict(smoothing_sigma_px=0.5, background_sigma_px=12.0, core_sigma_px=1.8)
output_dir = Path('DAPI_Core_Enhanced')
output_dir.mkdir(exist_ok=True)
enhanced_dapi_planes = {}
manifest = []
for name, source_path in plane_files.items():
    raw = raw_planes[name][dapi_channel_index].astype(np.float32)
    smooth = gaussian_filter(raw, settings['smoothing_sigma_px'])
    background = gaussian_filter(smooth, settings['background_sigma_px'])
    corrected = np.maximum(smooth - background, 0)
    enhanced = np.maximum(-gaussian_laplace(corrected, settings['core_sigma_px']) * settings['core_sigma_px']**2, 0).astype(np.float32)
    assert enhanced.shape == (1152, 2189) and np.isfinite(enhanced).all()
    enhanced_dapi_planes[name] = enhanced
    destination = output_dir / f'{name}_DAPI_core_enhanced.tif'
    tifffile.imwrite(destination, enhanced, imagej=True, photometric='minisblack',
        resolution=(1/pixel_size_um, 1/pixel_size_um),
        metadata={'axes': 'YX', 'unit': 'micron', 'Info': f'LoG core response from C3 of {source_path.name}; settings {settings}'})
    with tifffile.TiffFile(destination) as saved:
        assert np.array_equal(saved.asarray(), enhanced)
    manifest.append({'plane': name, 'source': source_path.name, 'channel': 'C3 DAPI',
                     'output': destination.name, 'shape_yx': list(enhanced.shape), 'dtype': 'float32'})
    print(f'{name}: saved and verified {destination}; range {enhanced.min():.3f} to {enhanced.max():.3f}')
(output_dir / 'enhancement_settings.json').write_text(json.dumps({
    'settings': settings, 'xy_pixel_size_um': pixel_size_um,
    'algorithm': 'Gaussian smoothing; subtract Gaussian background; clip at zero; scale-normalized negative LoG; clip at zero',
    'boundary_mode': 'reflect (scipy default)', 'planes': manifest,
    'purpose': 'Candidate nuclear segmentation input; core response, not nuclear boundaries or raw fluorescence',
    'raw_channels_modified': False}, indent=2))
print('Complete: three enhanced DAPI TIFFs and enhancement_settings.json. No segmentation performed.')


# %% Cell 12

