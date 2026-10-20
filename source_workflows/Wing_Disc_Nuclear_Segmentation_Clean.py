# Exported notebook workflow: Wing_Disc_Nuclear_Segmentation_Clean.ipynb

# %% Cell 1
# Diagnostic 1: fresh kernel, software, GPU, and cached model.
from pathlib import Path
import sys, json, inspect
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import tensorflow as tf
import deepcell
from deepcell.applications import NuclearSegmentation

project_dir = Path.cwd()
pixel_size_um = 0.5546006944444444
print('Project:', project_dir)
print('Python:', sys.version.split()[0])
print('DeepCell:', deepcell.__version__, '; TensorFlow:', tf.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))
print('NuclearSegmentation constructor:', inspect.signature(NuclearSegmentation))
print('Predict API:', inspect.signature(NuclearSegmentation.predict))
model_cache = Path.home() / '.deepcell' / 'models'
print('Cached model files:', [p.name for p in model_cache.glob('*Nuclear*')])
print('Fresh diagnostic session. No model loaded or segmentation run.')


# %% Cell 2
# Diagnostic 2: validate the three saved planes and inspect middle crop #2.
plane_files = {'top': 'top_Z28-Z29_4channels.tif',
               'middle': 'middle_Z33-Z34_4channels.tif',
               'bottom': 'bottom_Z38-Z39_4channels.tif'}
raw_planes, core_planes = {}, {}
settings = json.loads((project_dir / 'DAPI_Core_Enhanced' / 'enhancement_settings.json').read_text())
print('Saved enhancement settings:', settings['settings'])
for plane, filename in plane_files.items():
    with tifffile.TiffFile(project_dir / filename) as tif:
        raw = tif.asarray()
        axes = tif.series[0].axes
    core = tifffile.imread(project_dir / 'DAPI_Core_Enhanced' / f'{plane}_DAPI_core_enhanced.tif')
    assert axes == 'CYX' and raw.shape == (4, 1152, 2189)
    assert raw.dtype == np.uint16
    assert core.shape == raw.shape[1:] and np.isfinite(core).all() and core.min() >= 0
    raw_planes[plane], core_planes[plane] = raw, core
    print(plane, 'raw:', raw.shape, raw.dtype, axes,
          '; core:', core.shape, core.dtype, '; range:', float(core.min()), float(core.max()))

x0, y0, x1, y1 = 1004, 504, 1196, 696
raw_crop = raw_planes['middle'][2, y0:y1, x0:x1]
core_crop = core_planes['middle'][y0:y1, x0:x1]
print('Pixel size:', pixel_size_um, 'um/pixel; 3 um nucleus diameter:', 3/pixel_size_um, 'pixels')
print('Core crop percentiles [0, 50, 99, 99.8, 100]:', np.percentile(core_crop, [0, 50, 99, 99.8, 100]))
print('Core crop positive pixels:', 100*np.mean(core_crop > 0), '%')
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=130)
for ax, image, title in zip(axes, [raw_crop, core_crop], ['Middle crop #2: raw C3 DAPI', 'Middle crop #2: saved core-enhanced DAPI']):
    lo, hi = np.percentile(image, [1, 99.8])
    ax.imshow(image, cmap='gray', vmin=lo, vmax=max(hi, lo+1e-6), interpolation='nearest')
    ax.set_title(title)
    ax.plot([10, 10+10/pixel_size_um], [177, 177], color='yellow', linewidth=3)
    ax.text(10, 170, '10 um', color='yellow')
    ax.axis('off')
plt.tight_layout()
plt.show()
print('Intended nuclear input: saved core-enhanced C3 DAPI only.')
print('Diagnostic display only. No segmentation, resizing, or export performed.')


# %% Cell 3
# Diagnostic 3: load the cached nuclear model and inspect its settings.
# No prediction or mask generation in this cell.
if 'nuclear_app' not in globals():
    nuclear_app = NuclearSegmentation.from_version('1.1')

print('Model loaded:', type(nuclear_app).__name__)
print('Model input shape:', nuclear_app.model.input_shape)
print('Model output shapes:', nuclear_app.model.output_shape)
print('Model pixel size:', nuclear_app.model_mpp, 'um/pixel')
print('Image pixel size:', pixel_size_um, 'um/pixel')
print('Configured postprocessing:', nuclear_app.postprocess_kwargs)
print('Preprocessing function:', nuclear_app.preprocessing_fn.__name__)
print('Preprocessing signature:', inspect.signature(nuclear_app.preprocessing_fn))
print('Postprocessing function:', nuclear_app.postprocessing_fn.__name__)
print('Postprocessing signature:', inspect.signature(nuclear_app.postprocessing_fn))

# Check the precise intended input, without feeding it to the model yet.
padding_px = 32
core_input_patch = core_planes['middle'][
    y0-padding_px:y1+padding_px, x0-padding_px:x1+padding_px].astype(np.float32)
deepcell_input = core_input_patch[None, ..., None]
assert deepcell_input.shape == (1, 256, 256, 1)
assert np.isfinite(deepcell_input).all() and deepcell_input.min() >= 0
print('Prepared input:', deepcell_input.shape, deepcell_input.dtype)
print('Nuclear channel: saved core-enhanced C3 DAPI, middle crop #2 + 32 px context.')
print('No prediction, resizing experiment, or mask export performed.')


# %% Cell 4
# Baseline segmentation: middle crop #2, saved core-enhanced DAPI only.
# Default DeepCell settings; inspect before changing parameters. No export.
from skimage.segmentation import find_boundaries

baseline_settings = dict(nuclear_app.postprocess_kwargs)
print('Nuclear input:', deepcell_input.shape, deepcell_input.dtype, flush=True)
print('Image calibration:', pixel_size_um, 'um/pixel', flush=True)
print('Baseline settings:', baseline_settings, flush=True)
baseline_prediction = nuclear_app.predict(
    deepcell_input.copy(), image_mpp=pixel_size_um,
    postprocess_kwargs=baseline_settings)
assert baseline_prediction.shape == deepcell_input.shape
p = padding_px
baseline_crop_labels = baseline_prediction[0, p:-p, p:-p, 0].astype(np.int32)
assert baseline_crop_labels.shape == core_crop.shape

# Identical display contrast in both panels; cyan marks mask boundaries.
display_hi = max(float(np.percentile(core_crop, 99.8)), 1e-6)
display_gray = np.clip(core_crop / display_hi, 0, 1)
baseline_overlay = np.repeat(display_gray[..., None], 3, axis=-1)
baseline_overlay[find_boundaries(baseline_crop_labels, mode='inner')] = (0, 1, 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=130)
axes[0].imshow(display_gray, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
axes[1].imshow(baseline_overlay, interpolation='nearest')
axes[0].set_title('Middle crop #2 — core-enhanced DAPI')
axes[1].set_title('DeepCell baseline — cyan mask boundaries')
for ax in axes:
    ax.plot([10, 10+10/pixel_size_um], [177, 177], color='yellow', linewidth=3)
    ax.text(10, 170, '10 um', color='yellow')
    ax.axis('off')
plt.tight_layout()
plt.show()
label_ids, pixel_counts = np.unique(baseline_crop_labels, return_counts=True)
visible_areas = pixel_counts[label_ids > 0]
print('Nuclei intersecting original crop:', len(visible_areas))
print('Nuclei in full padded patch:', np.count_nonzero(np.unique(baseline_prediction)))
if len(visible_areas):
    print('Median visible mask area:', float(np.median(visible_areas)), 'pixels')
print('Crop-edge masks may be partial. No area filtering or mask export.')


# %% Cell 5
# Exploratory size trial: 3x resampling of the SAME core-enhanced input.
# Deliberately present enlarged pixels at model scale; not a new calibration.
from skimage.transform import resize

scale_factor = 3
scaled_shape = tuple(n * scale_factor for n in core_input_patch.shape)
scaled_patch = resize(core_input_patch, scaled_shape, order=1,
                      preserve_range=True, anti_aliasing=False).astype(np.float32)
scaled_input = scaled_patch[None, ..., None]
true_scaled_mpp = pixel_size_um / scale_factor
print('Resampling factor:', scale_factor, '; enlarged input:', scaled_input.shape, flush=True)
print('Original physical calibration:', pixel_size_um, 'um/pixel')
print('Physical pixel size after resampling:', true_scaled_mpp, 'um/pixel')
print('EXPERIMENT: image_mpp=model_mpp prevents automatic resizing from undoing enlargement.', flush=True)
print('This is a model-size experiment, not a physical calibration change.', flush=True)
scaled_prediction = nuclear_app.predict(
    scaled_input.copy(), image_mpp=nuclear_app.model_mpp,
    postprocess_kwargs=dict(baseline_settings))
assert scaled_prediction.shape == scaled_input.shape
scaled_labels = scaled_prediction[0, ..., 0]

# Nearest-neighbor mapping preserves integer label IDs at original resolution.
size_trial_patch_labels = resize(
    scaled_labels, core_input_patch.shape, order=0,
    preserve_range=True, anti_aliasing=False).astype(np.int32)
p = padding_px
size_trial_crop_labels = size_trial_patch_labels[p:-p, p:-p]
assert size_trial_crop_labels.shape == core_crop.shape
size_trial_overlay = np.repeat(display_gray[..., None], 3, axis=-1)
size_trial_overlay[find_boundaries(size_trial_crop_labels, mode='inner')] = (0, 1, 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=130)
axes[0].imshow(display_gray, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
axes[1].imshow(size_trial_overlay, interpolation='nearest')
axes[0].set_title('Middle crop #2 — core-enhanced DAPI')
axes[1].set_title('DeepCell 3x size trial — cyan boundaries')
for ax in axes:
    ax.plot([10, 10+10/pixel_size_um], [177, 177], color='yellow', linewidth=3)
    ax.text(10, 170, '10 um', color='yellow')
    ax.axis('off')
plt.tight_layout()
plt.show()
ids, counts = np.unique(size_trial_crop_labels, return_counts=True)
areas = counts[ids > 0]
print('Baseline crop nuclei:', len(visible_areas))
print('Size-trial crop nuclei:', len(areas))
if len(areas):
    print('Median visible mask area at original resolution:', float(np.median(areas)), 'pixels')
print('Inspect missing nuclei, merged nuclei, and splits. Crop-edge masks may be partial.')
print('Original files unchanged. No mask filtering or export.')


# %% Cell 6
# Close-up inspection only: existing 3x size-trial masks, no new segmentation.
# Central 96 x 96 pixels within middle crop #2 (~53.2 um across).
close_x0, close_y0, close_x1, close_y1 = 48, 48, 144, 144
close_slice = (slice(close_y0, close_y1), slice(close_x0, close_x1))
close_gray = display_gray[close_slice]
# Crop the full overlay to avoid inventing boundaries at the inspection edges.
close_overlay = size_trial_overlay[close_slice]
close_labels = size_trial_crop_labels[close_slice]

fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=160)
axes[0].imshow(close_gray, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
axes[1].imshow(close_overlay, interpolation='nearest')
axes[0].set_title('Middle crop #2 — central close-up, core-enhanced DAPI')
axes[1].set_title('Same region — existing DeepCell mask boundaries')
for ax in axes:
    ax.plot([5, 5+5/pixel_size_um], [88, 88], color='yellow', linewidth=3)
    ax.text(5, 82, '5 um', color='yellow')
    ax.axis('off')
plt.tight_layout()
plt.show()
print('Full-plane coordinates: X', x0+close_x0, 'to', x0+close_x1,
      '; Y', y0+close_y0, 'to', y0+close_y1, '(upper bounds exclusive)')
print('Masks intersecting close-up:', np.count_nonzero(np.unique(close_labels)))
print('Inspect: one mask containing multiple cores; multiple masks within one nucleus; weak cores without masks.')
print('Masks crossing the close-up edges are partial. Display only; no rerun or export.')


# %% Cell 7
# Full middle-plane trial: approved 3x size experiment and default settings.
# DeepCell tiles inference internally. No export; review the overlay first.
import time
from skimage.transform import resize
from skimage.segmentation import find_boundaries

middle_scale_factor = 3
middle_core = core_planes['middle'].astype(np.float32, copy=False)
assert middle_core.shape == (1152, 2189)
middle_scaled_shape = tuple(n * middle_scale_factor for n in middle_core.shape)
print('Original middle plane:', middle_core.shape, flush=True)
print('Enlarged input:', middle_scaled_shape, flush=True)
print('Same size-trial convention: image_mpp=model_mpp; original calibration retained for output.', flush=True)
print('Settings:', baseline_settings, flush=True)
started = time.perf_counter()
middle_scaled_image = resize(
    middle_core, middle_scaled_shape, order=1,
    preserve_range=True, anti_aliasing=False).astype(np.float32)
print('Input prepared. Starting full-plane prediction...', flush=True)
middle_scaled_prediction = nuclear_app.predict(
    middle_scaled_image[None, ..., None], batch_size=4,
    image_mpp=nuclear_app.model_mpp,
    postprocess_kwargs=dict(baseline_settings))
assert middle_scaled_prediction.shape == (1, *middle_scaled_shape, 1)
print('Prediction complete. Mapping labels to original pixels...', flush=True)
middle_nuclear_labels = resize(
    middle_scaled_prediction[0, ..., 0], middle_core.shape,
    order=0, preserve_range=True, anti_aliasing=False).astype(np.uint32)
assert middle_nuclear_labels.shape == middle_core.shape
# Release enlarged arrays after mapping; keep the original-resolution labels.
del middle_scaled_image, middle_scaled_prediction
print('Elapsed:', round(time.perf_counter()-started, 1), 'seconds', flush=True)
middle_ids = np.unique(middle_nuclear_labels)
print('Nuclei in middle plane:', np.count_nonzero(middle_ids), flush=True)
print('Mask dimensions:', middle_nuclear_labels.shape, '; dtype:', middle_nuclear_labels.dtype)

middle_display_hi = max(float(np.percentile(middle_core, 99.8)), 1e-6)
middle_gray = np.clip(middle_core / middle_display_hi, 0, 1)
middle_overlay = np.repeat(middle_gray[..., None], 3, axis=-1)
middle_overlay[find_boundaries(middle_nuclear_labels, mode='inner')] = (0, 1, 1)
fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=150)
axes[0].imshow(middle_gray, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
axes[1].imshow(middle_overlay, interpolation='nearest')
axes[0].set_title('Entire middle plane — core-enhanced DAPI')
axes[1].set_title('DeepCell 3x size trial — nuclear boundaries')
for ax in axes:
    ax.plot([60, 60+50/pixel_size_um], [1090, 1090], color='yellow', linewidth=3)
    ax.text(60, 1060, '50 um', color='yellow')
    ax.axis('off')
plt.tight_layout()
plt.show()
print('Full-plane preview may hide small errors; inspect close-ups before accepting masks.')
print('Masks remain in memory. No export or area filtering. Top and bottom not processed.')


# %% Cell 8
# Compare the SAME crop: crop-only segmentation versus full-plane segmentation.
# After reconnecting, rerun the cells above first to restore in-memory results.
required = ['size_trial_crop_labels', 'middle_nuclear_labels', 'core_crop', 'baseline_settings']
missing = [name for name in required if name not in globals()]
assert not missing, 'Rerun the preceding cells first; missing: ' + ', '.join(missing)
from skimage.segmentation import find_boundaries

full_plane_crop_labels = middle_nuclear_labels[y0:y1, x0:x1]
assert full_plane_crop_labels.shape == size_trial_crop_labels.shape == core_crop.shape
comparison_hi = max(float(np.percentile(core_crop, 99.8)), 1e-6)
comparison_gray = np.clip(core_crop / comparison_hi, 0, 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=160)
for ax, labels, title in zip(axes,
    [size_trial_crop_labels, full_plane_crop_labels],
    ['Crop-only segmentation: middle crop #2', 'Full-plane segmentation: SAME crop #2']):
    overlay = np.repeat(comparison_gray[..., None], 3, axis=-1)
    overlay[find_boundaries(labels, mode='inner')] = (0, 1, 1)
    ax.imshow(overlay, interpolation='nearest')
    ax.set_title(title)
    ax.plot([10, 10+10/pixel_size_um], [177, 177], color='yellow', linewidth=3)
    ax.text(10, 170, '10 um', color='yellow')
    ax.axis('off')
    ids, counts = np.unique(labels, return_counts=True)
    areas = counts[ids > 0]
    print(title, '; objects:', len(areas),
          '; median visible area:', float(np.median(areas)) if len(areas) else 0,
          '; labeled-pixel coverage:', round(100*np.mean(labels > 0), 1), '%')
plt.tight_layout()
plt.show()
print('Same coordinates, same core-enhanced image, same contrast. Label IDs need not match.')
print('Compare missing cores, merges, splits, and mask extent. No new prediction or export.')


# %% Cell 9
# Tissue-outline preview from RAW middle-plane C3 DAPI.
# This does not change nuclear labels or remove any detections.
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_objects
from skimage.segmentation import find_boundaries

# Exploratory tissue-scale parameters, independent of nuclear segmentation.
tissue_smoothing_sigma_px = 12.0
tissue_threshold_multiplier = 0.7
tissue_closing_radius_px = 12
tissue_min_component_area_px = 5000

middle_raw_dapi = raw_planes['middle'][2].astype(np.float32)
tissue_signal = gaussian_filter(middle_raw_dapi, tissue_smoothing_sigma_px)
tissue_otsu = float(threshold_otsu(tissue_signal))
tissue_threshold = tissue_threshold_multiplier * tissue_otsu
tissue_candidate = tissue_signal > tissue_threshold
tissue_candidate = binary_closing(tissue_candidate, footprint=disk(tissue_closing_radius_px))
tissue_candidate = remove_small_objects(tissue_candidate, min_size=tissue_min_component_area_px)
tissue_mask_preview = binary_fill_holes(tissue_candidate)
assert tissue_mask_preview.shape == middle_nuclear_labels.shape

raw_lo, raw_hi = np.percentile(middle_raw_dapi, [1, 99.8])
tissue_display_gray = np.clip((middle_raw_dapi-raw_lo)/max(raw_hi-raw_lo, 1e-6), 0, 1)
fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=150)
axes[0].imshow(tissue_display_gray, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
axes[1].imshow(tissue_display_gray, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
if tissue_mask_preview.any() and not tissue_mask_preview.all():
    axes[1].contour(tissue_mask_preview.astype(np.uint8), levels=[0.5], colors=['yellow'], linewidths=1)
axes[0].set_title('Middle plane — raw C3 DAPI')
axes[1].set_title('Candidate tissue outline — yellow; review before filtering')
for ax in axes:
    ax.plot([60, 60+50/pixel_size_um], [1090, 1090], color='cyan', linewidth=3)
    ax.text(60, 1060, '50 um', color='cyan')
    ax.axis('off')
plt.tight_layout()
plt.show()
print('Tissue smoothing sigma:', tissue_smoothing_sigma_px, 'px')
print('Otsu threshold:', tissue_otsu, '; applied threshold:', tissue_threshold)
print('Tissue coverage:', round(100*np.mean(tissue_mask_preview), 2), '% of image')
print('Check that the outline includes dim tissue and disc edges, while excluding empty background.')
print('Preview only: all nuclear labels retained; no mask filtering or export.')


# %% Cell 10
# Revised tissue outline: include the dim connection between tissue regions.
# Preview only. Nuclear labels are unchanged.
from scipy.ndimage import label as connected_components

tissue_threshold_multiplier_revised = 0.5  # previously 0.7: include dimmer tissue
tissue_closing_radius_px_revised = 32       # previously 12: bridge small gaps
revised_threshold = tissue_threshold_multiplier_revised * tissue_otsu
revised_candidate = tissue_signal > revised_threshold
revised_candidate = binary_closing(
    revised_candidate, footprint=disk(tissue_closing_radius_px_revised))
revised_candidate = remove_small_objects(
    revised_candidate, min_size=tissue_min_component_area_px)
tissue_mask_revised = binary_fill_holes(revised_candidate)
components, component_count = connected_components(tissue_mask_revised, structure=np.ones((3, 3)))
component_areas = np.bincount(components.ravel())[1:]
print('Connected tissue components:', component_count)
print('Component areas:', sorted(component_areas.tolist(), reverse=True))
print('Revised threshold:', revised_threshold, '; closing radius:', tissue_closing_radius_px_revised, 'px')
print('Tissue coverage:', round(100*np.mean(tissue_mask_revised), 2), '%')

fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=180)
for ax, mask, title in zip(axes,
    [tissue_mask_preview, tissue_mask_revised],
    ['Previous tissue outline', 'Revised outline — dim connection included']):
    ax.imshow(tissue_display_gray, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    if mask.any() and not mask.all():
        ax.contour(mask.astype(np.uint8), levels=[0.5], colors=['yellow'], linewidths=1)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
print('Confirm the connection and check for excessive background inclusion.')
print('Revised mask remains a candidate; no nuclear filtering or export performed.')


# %% Cell 11
# Apply approved tissue outline to existing middle-plane nuclei.
# Retain a whole nucleus if at least 50% of its mask lies inside tissue.
# This preserves mask shapes at the disc edge; no segmentation rerun or export.
assert tissue_mask_revised.shape == middle_nuclear_labels.shape
middle_tissue_mask = tissue_mask_revised.copy()
minimum_tissue_overlap = 0.5
labels_flat = middle_nuclear_labels.ravel().astype(np.int64)
total_pixels_per_id = np.bincount(labels_flat)
inside_pixels_per_id = np.bincount(
    labels_flat[middle_tissue_mask.ravel()], minlength=len(total_pixels_per_id))
overlap_per_id = np.divide(inside_pixels_per_id, total_pixels_per_id,
    out=np.zeros(len(total_pixels_per_id), dtype=np.float64), where=total_pixels_per_id > 0)
keep_id = (total_pixels_per_id > 0) & (overlap_per_id >= minimum_tissue_overlap)
keep_id[0] = False
middle_filtered_nuclear_labels = np.where(
    keep_id[middle_nuclear_labels], middle_nuclear_labels, 0).astype(np.uint32)
original_count = int(np.count_nonzero(total_pixels_per_id[1:]))
retained_count = int(np.count_nonzero(keep_id))
print('Original objects:', original_count)
print('Retained tissue nuclei:', retained_count)
print('Excluded objects:', original_count-retained_count)
print('Rule: at least', 100*minimum_tissue_overlap, '% of each mask inside approved tissue.')
assert np.all(middle_filtered_nuclear_labels[middle_filtered_nuclear_labels > 0]
              == middle_nuclear_labels[middle_filtered_nuclear_labels > 0])

filtered_overlay = np.repeat(middle_gray[..., None], 3, axis=-1)
filtered_overlay[find_boundaries(middle_filtered_nuclear_labels, mode='inner')] = (0, 1, 1)
fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=180)
axes[0].imshow(middle_gray, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
axes[1].imshow(filtered_overlay, interpolation='nearest')
axes[0].set_title('Middle plane — core-enhanced DAPI')
axes[1].set_title(f'Tissue-filtered nuclei — {retained_count} masks')
axes[1].contour(middle_tissue_mask.astype(np.uint8), levels=[0.5], colors=['yellow'], linewidths=0.7)
for ax in axes:
    ax.plot([60, 60+50/pixel_size_um], [1090, 1090], color='yellow', linewidth=3)
    ax.text(60, 1060, '50 um', color='yellow')
    ax.axis('off')
plt.tight_layout()
plt.show()
print('Cyan = retained nuclear boundaries; yellow = approved tissue outline.')
print('Whole masks retained; some edge masks may extend slightly beyond the outline.')
print('Original segmentation retained separately. No fluorescence/size filtering or export.')


# %% Cell 12
# Boundary QC: flag weak local RAW-DAPI contrast; keep ALL masks.
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.measure import regionprops

boundary_band_px = 10  # ~5.5 um on either side of the tissue outline
ring_inner_px, ring_outer_px = 2, 5
minimum_local_ratio = 1.2  # exploratory flag: nucleus median < 1.2 x local background
qc_labels = middle_filtered_nuclear_labels
qc_raw = raw_planes['middle'][2].astype(np.float32)
tissue_edge = find_boundaries(middle_tissue_mask, mode='thick')
near_edge = distance_transform_edt(~tissue_edge) <= boundary_band_px
boundary_ids = np.unique(qc_labels[near_edge])
boundary_ids = set(boundary_ids[boundary_ids > 0].tolist())
boundary_qc = []
for region in regionprops(qc_labels):
    if region.label not in boundary_ids:
        continue
    ry0, rx0, ry1, rx1 = region.bbox
    ya, xa = max(0, ry0-ring_outer_px), max(0, rx0-ring_outer_px)
    yb, xb = min(qc_raw.shape[0], ry1+ring_outer_px), min(qc_raw.shape[1], rx1+ring_outer_px)
    local_labels = qc_labels[ya:yb, xa:xb]
    nucleus = local_labels == region.label
    # Exclude all detected nuclei, including those rejected by tissue filtering.
    ring = (binary_dilation(nucleus, structure=disk(ring_outer_px)) &
            ~binary_dilation(nucleus, structure=disk(ring_inner_px)) &
            (middle_nuclear_labels[ya:yb, xa:xb] == 0))
    if np.count_nonzero(ring) < 10:
        boundary_qc.append(dict(label=region.label, center=region.centroid,
                                ratio=np.nan, flagged=False, evaluable=False))
        continue
    local_raw = qc_raw[ya:yb, xa:xb]
    signal = float(np.median(local_raw[nucleus]))
    background = float(np.median(local_raw[ring]))
    ratio = signal / max(background, 1.0)
    boundary_qc.append(dict(label=region.label, center=region.centroid,
                            ratio=ratio, flagged=ratio < minimum_local_ratio, evaluable=True))
flagged_records = [r for r in boundary_qc if r['flagged']]
flagged_ids = [r['label'] for r in flagged_records]
flag_lookup = np.zeros(int(qc_labels.max())+1, dtype=bool)
flag_lookup[flagged_ids] = True
print('Boundary masks checked:', len(boundary_qc))
print('Masks with insufficient background pixels:', sum(not r['evaluable'] for r in boundary_qc))
print('Flagged for review:', len(flagged_ids), '; exploratory median-DAPI ratio <', minimum_local_ratio)

# Select one 192px close-up containing the most flagged mask centers.
# If none flagged, show a boundary region instead.
records_for_view = flagged_records or boundary_qc
assert records_for_view, 'No boundary masks found.'
centers = np.array([r['center'] for r in records_for_view])
scores = [np.sum(np.all(np.abs(centers-c) < 80, axis=1)) for c in centers]
cy, cx = centers[int(np.argmax(scores))]
qc_y0 = int(np.clip(round(cy)-96, 0, qc_raw.shape[0]-192))
qc_x0 = int(np.clip(round(cx)-96, 0, qc_raw.shape[1]-192))
qc_slice = (slice(qc_y0, qc_y0+192), slice(qc_x0, qc_x0+192))
view_raw = qc_raw[qc_slice]
lo, hi = np.percentile(view_raw, [1, 99.8])
view_gray = np.clip((view_raw-lo)/max(hi-lo, 1e-6), 0, 1)
view_overlay = np.repeat(view_gray[..., None], 3, axis=-1)
full_boundaries = find_boundaries(qc_labels, mode='inner')
view_boundaries = full_boundaries[qc_slice]
view_flags = flag_lookup[qc_labels[qc_slice]]
view_overlay[view_boundaries] = (0, 1, 1)
view_overlay[view_boundaries & view_flags] = (1, 0, 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=160)
axes[0].imshow(view_gray, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
axes[1].imshow(view_overlay, interpolation='nearest')
axes[0].set_title('Boundary close-up — RAW C3 DAPI')
axes[1].set_title('Magenta = weak-contrast flags; cyan = other masks')
for ax in axes:
    ax.plot([10, 10+10/pixel_size_um], [177,177], color='yellow', linewidth=3)
    ax.text(10,170,'10 um',color='yellow')
    ax.axis('off')
plt.tight_layout()
plt.show()
print('Close-up bounds: X', qc_x0, 'to', qc_x0+192, '; Y', qc_y0, 'to', qc_y0+192)
print('Flagged masks in view:', np.count_nonzero(np.unique(qc_labels[qc_slice][view_flags])))
print('Flags are candidates, not confirmed errors. Dim nuclei and crowded regions need review.')
print('All masks unchanged. No deletion, filtering, or export.')


# %% Cell 13
# Exclude the reviewed magenta boundary flags from a NEW mask array.
# Previous arrays and the QC records remain available; no export yet.
assert np.array_equal(qc_labels, middle_filtered_nuclear_labels)
reviewed_excluded_ids = np.array(flagged_ids, dtype=np.uint32)
middle_final_nuclear_labels = middle_filtered_nuclear_labels.copy()
reviewed_excluded_pixels = flag_lookup[middle_final_nuclear_labels]
middle_final_nuclear_labels[reviewed_excluded_pixels] = 0
before_ids = np.unique(middle_filtered_nuclear_labels)
after_ids = np.unique(middle_final_nuclear_labels)
before_count = int(np.count_nonzero(before_ids))
after_count = int(np.count_nonzero(after_ids))
assert before_count-after_count == len(reviewed_excluded_ids)
print('Before boundary QC:', before_count)
print('Excluded reviewed magenta masks:', len(reviewed_excluded_ids))
print('Retained middle-plane nuclei:', after_count)

# Same close-up and raw-DAPI contrast as the review cell.
clean_view_labels = middle_final_nuclear_labels[qc_slice]
clean_view_overlay = np.repeat(view_gray[..., None], 3, axis=-1)
clean_view_boundaries = find_boundaries(middle_final_nuclear_labels, mode='inner')[qc_slice]
clean_view_overlay[clean_view_boundaries] = (0, 1, 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=160)
axes[0].imshow(view_overlay, interpolation='nearest')
axes[1].imshow(clean_view_overlay, interpolation='nearest')
axes[0].set_title('Before — magenta masks marked for exclusion')
axes[1].set_title('After — retained masks in cyan')
for ax in axes:
    ax.plot([10, 10+10/pixel_size_um], [177,177], color='yellow', linewidth=3)
    ax.text(10,170,'10 um',color='yellow')
    ax.axis('off')
plt.tight_layout()
plt.show()
print('Previous masks preserved separately. Retained label IDs and mask shapes unchanged.')
print('No files exported. QC records retain excluded IDs and local contrast measurements.')


# %% Cell 14
# Entire middle-plane result after approved tissue and boundary filtering.
# These filters already operated globally; this cell displays the full result.
assert middle_final_nuclear_labels.shape == (1152, 2189)
final_middle_overlay = np.repeat(middle_gray[..., None], 3, axis=-1)
final_middle_overlay[find_boundaries(middle_final_nuclear_labels, mode='inner')] = (0, 1, 1)
fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=180)
axes[0].imshow(middle_gray, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
axes[1].imshow(final_middle_overlay, interpolation='nearest')
axes[0].set_title('Entire middle plane — core-enhanced DAPI')
axes[1].set_title('Entire middle plane — final retained nuclear masks')
for ax in axes:
    ax.plot([60, 60+50/pixel_size_um], [1090,1090], color='yellow', linewidth=3)
    ax.text(60,1060,'50 um',color='yellow')
    ax.axis('off')
plt.tight_layout()
plt.show()
print('Whole-plane dimensions:', middle_final_nuclear_labels.shape)
print('Retained nuclei:', np.count_nonzero(np.unique(middle_final_nuclear_labels)))
print('Boundary-QC exclusions across the entire tissue:', len(reviewed_excluded_ids))
print('Boundary QC was applied around the full tissue outline, not only the displayed close-up.')
print('No prediction rerun or export. Top and bottom remain unprocessed.')


# %% Cell 15
# Approved pipeline for all three planes. Run sequentially to limit memory.
# Middle reuses its reviewed final labels; top/bottom use identical settings.
import time
from datetime import datetime
from scipy.ndimage import gaussian_filter, binary_fill_holes, distance_transform_edt, binary_dilation
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_objects
from skimage.measure import regionprops
from skimage.transform import resize
from skimage.segmentation import find_boundaries

pipeline_config = dict(scale_factor=3, tissue_smoothing_sigma_px=12.0,
    tissue_threshold_multiplier=0.5, tissue_closing_radius_px=32,
    tissue_min_component_area_px=5000, minimum_tissue_overlap=0.5,
    boundary_band_px=10, ring_inner_px=2, ring_outer_px=5,
    minimum_local_ratio=1.2, pixel_size_um=pixel_size_um,
    model_mpp=float(nuclear_app.model_mpp), postprocessing=dict(baseline_settings))
output_dir = project_dir / ('Nuclear_Masks_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
output_dir.mkdir(exist_ok=False)
final_plane_masks, final_tissue_masks, plane_summaries = {}, {}, {}

def segment_and_filter_plane(plane):
    core = core_planes[plane]
    raw = raw_planes[plane][2].astype(np.float32)
    enlarged_shape = tuple(3*n for n in core.shape)
    enlarged = resize(core, enlarged_shape, order=1, preserve_range=True,
                      anti_aliasing=False).astype(np.float32)
    prediction = nuclear_app.predict(enlarged[None,...,None], batch_size=4,
        image_mpp=nuclear_app.model_mpp, postprocess_kwargs=dict(baseline_settings))
    labels = resize(prediction[0,...,0], core.shape, order=0,
        preserve_range=True, anti_aliasing=False).astype(np.uint32)
    del enlarged, prediction
    signal = gaussian_filter(raw, 12.0)
    applied_threshold = 0.5*float(threshold_otsu(signal))
    tissue = binary_closing(signal > applied_threshold, footprint=disk(32))
    tissue = binary_fill_holes(remove_small_objects(tissue, min_size=5000))
    flat = labels.ravel().astype(np.int64)
    totals = np.bincount(flat)
    inside = np.bincount(flat[tissue.ravel()], minlength=len(totals))
    overlaps = np.divide(inside, totals, out=np.zeros(len(totals)), where=totals>0)
    keep = (totals>0) & (overlaps>=0.5)
    keep[0] = False
    tissue_labels = np.where(keep[labels], labels, 0).astype(np.uint32)
    edge = find_boundaries(tissue, mode='thick')
    band = distance_transform_edt(~edge)<=10
    boundary_ids = set(np.unique(tissue_labels[band]).tolist()) - {0}
    qc_records = []
    for region in regionprops(tissue_labels):
        if region.label not in boundary_ids:
            continue
        by0,bx0,by1,bx1 = region.bbox
        ya,xa = max(0,by0-5),max(0,bx0-5)
        yb,xb = min(raw.shape[0],by1+5),min(raw.shape[1],bx1+5)
        nucleus = tissue_labels[ya:yb,xa:xb]==region.label
        ring = (binary_dilation(nucleus, structure=disk(5)) &
                ~binary_dilation(nucleus, structure=disk(2)) & (labels[ya:yb,xa:xb]==0))
        evaluable = np.count_nonzero(ring)>=10
        ratio = None
        if evaluable:
            patch = raw[ya:yb,xa:xb]
            ratio = float(np.median(patch[nucleus])) / max(float(np.median(patch[ring])),1.0)
        qc_records.append(dict(label=int(region.label), ratio=ratio,
            evaluable=bool(evaluable), flagged=bool(evaluable and ratio<1.2)))
    excluded = [r['label'] for r in qc_records if r['flagged']]
    exclude_lookup = np.zeros(len(totals), dtype=bool)
    exclude_lookup[excluded] = True
    final = tissue_labels.copy()
    final[exclude_lookup[final]] = 0
    summary = dict(initial_objects=int(np.count_nonzero(totals[1:])),
        after_tissue=int(np.count_nonzero(keep)), boundary_excluded=len(excluded),
        final_nuclei=int(np.count_nonzero(np.unique(final))),
        tissue_coverage=float(np.mean(tissue)), tissue_threshold=applied_threshold)
    return final, tissue, summary, qc_records

for plane in ['top','middle','bottom']:
    started = time.perf_counter()
    print('Processing', plane, '...', flush=True)
    if plane=='middle':
        final = middle_final_nuclear_labels.copy()
        tissue = middle_tissue_mask.copy()
        summary = dict(initial_objects=32347, after_tissue=7445,
            boundary_excluded=len(reviewed_excluded_ids),
            final_nuclei=int(np.count_nonzero(np.unique(final))),
            tissue_coverage=float(np.mean(tissue)), tissue_threshold=float(revised_threshold),
            reused_reviewed_result=True)
        qc_records = [dict(label=int(r['label']),
            ratio=float(r['ratio']) if np.isfinite(r['ratio']) else None,
            evaluable=bool(r['evaluable']), flagged=bool(r['flagged'])) for r in boundary_qc]
    else:
        final,tissue,summary,qc_records = segment_and_filter_plane(plane)
    assert final.shape==(1152,2189) and final.dtype==np.uint32
    final_plane_masks[plane],final_tissue_masks[plane] = final,tissue
    plane_summaries[plane] = summary
    mask_path = output_dir / f'{plane}_nuclear_labels.tif'
    tifffile.imwrite(mask_path, final, photometric='minisblack',
        metadata={'axes':'YX','PhysicalSizeX':pixel_size_um,'PhysicalSizeY':pixel_size_um,
                  'PhysicalSizeXUnit':'µm','PhysicalSizeYUnit':'µm'},
        resolution=(1/pixel_size_um,1/pixel_size_um), resolutionunit='NONE')
    tifffile.imwrite(output_dir/f'{plane}_tissue_mask.tif', tissue.astype(np.uint8),
        photometric='minisblack', metadata={'axes':'YX'})
    assert np.array_equal(tifffile.imread(mask_path),final)
    (output_dir/f'{plane}_boundary_qc.json').write_text(json.dumps(qc_records,indent=2))
    print(plane, summary, '; elapsed:', round(time.perf_counter()-started,1),'s', flush=True)
    core = core_planes[plane]
    hi = max(float(np.percentile(core,99.8)),1e-6)
    gray = np.clip(core/hi,0,1)
    overlay = np.repeat(gray[...,None],3,axis=-1)
    overlay[find_boundaries(final,mode='inner')] = (0,1,1)
    fig,axes = plt.subplots(1,2,figsize=(18,6),dpi=180)
    axes[0].imshow(gray,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
    axes[1].imshow(overlay,interpolation='nearest')
    axes[0].set_title(f'{plane} — core-enhanced DAPI')
    axes[1].set_title(f'{plane} — {summary["final_nuclei"]} retained nuclear masks')
    for ax in axes:
        ax.axis('off')
        ax.plot([60,60+50/pixel_size_um],[1090,1090],color='yellow',linewidth=3)
        ax.text(60,1060,'50 um',color='yellow')
    plt.tight_layout()
    fig.savefig(output_dir/f'{plane}_masked_plane_preview.png',dpi=180,bbox_inches='tight')
    plt.show()
    plt.close(fig)

(output_dir/'pipeline_settings_and_counts.json').write_text(json.dumps(
    dict(settings=pipeline_config, summaries=plane_summaries,
         nuclear_input='saved core-enhanced C3 DAPI',
         scale_note='3x resampling, image_mpp=model_mpp for experimental apparent nuclear size; output at original physical calibration'),indent=2))
print('COMPLETE. Saved and read-back verified all 3 nuclear-label TIFFs:',output_dir,flush=True)
print('Label 0 is background. Each positive label identifies a nucleus; IDs need not be consecutive.')


# %% Cell 16
# Recover batch status after browser disconnection; no segmentation rerun.
print('Batch output folder:', output_dir)
print('Completed plane summaries:', plane_summaries)
print('Saved completion manifest:', (output_dir/'pipeline_settings_and_counts.json').exists())
print('Saved previews:', sorted(p.name for p in output_dir.glob('*preview.png')))


# %% Cell 17
# Restore all three saved previews after the browser connection interruption.
from IPython.display import display, Image
for plane in ['top','middle','bottom']:
    saved_labels = tifffile.imread(output_dir/f'{plane}_nuclear_labels.tif')
    assert np.array_equal(saved_labels, final_plane_masks[plane])
    print(plane.upper(), '—', plane_summaries[plane]['final_nuclei'], 'retained nuclei', flush=True)
    display(Image(filename=str(output_dir/f'{plane}_masked_plane_preview.png'), width=1100))
print('All three saved masks verified. Output folder:', output_dir)


# %% Cell 18

