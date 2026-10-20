"""Executed steps from Guided_Tutorial.ipynb; run from the tutorial folder."""
import matplotlib
matplotlib.use("Agg")  # Save plots without opening desktop windows.
# %%
from pathlib import Path
import csv
import json
import sys

# Find this tutorial whether opened from its own folder or the project root.
candidates = [Path.cwd(), *Path.cwd().parents]
candidates += [p / 'Writeup/tutorial' for p in list(candidates)]
TUTORIAL = next(p for p in candidates
                if (p / 'data/crop_manifest.json').exists())
sys.path.insert(0, str(TUTORIAL / 'code'))
RESULTS = TUTORIAL / 'results'
RESULTS.mkdir(exist_ok=True)

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from measure_nuclei import measure_nuclei
from phenotype_analysis import (
    read_measurements, gate_nuclei, count_regions, write_csv,
)
from tutorial_plots import (
    plot_mask_qc, plot_marker_gate,
    plot_spatial_gate, plot_region_summary,
)

channel_names = ['C1 GFP', 'C2 unknown', 'C3 DAPI', 'C4 RFP']
pixel_size_um = 0.5546006944444444
print('Example channels:', ', '.join(channel_names))

# %%
raw = tifffile.imread(TUTORIAL / 'data/middle_crop_raw.tif')
labels = tifffile.imread(TUTORIAL / 'data/middle_crop_labels.tif')
assert raw.shape[1:] == labels.shape
assert raw.shape[0] == len(channel_names)

print('Raw:', raw.shape, raw.dtype)
print('Labels:', labels.shape, labels.dtype)
print('Complete nuclei:', len(np.unique(labels[labels > 0])))

fig = plot_mask_qc(raw[2], labels, pixel_size_um)
fig.savefig(RESULTS / 'mask_qc.png', dpi=180)
plt.show()

# %%
crop_rows = measure_nuclei(
    raw, labels, channel_names, pixel_size_um=pixel_size_um,
)
crop_rows = [dict(plane='middle_crop', **row) for row in crop_rows]
write_csv(RESULTS / 'crop_measurements.csv', crop_rows, list(crop_rows[0]))

print(f'Measured {len(crop_rows):,} complete nuclear masks')
print('mask_id  area_px  GFP_mean  DAPI_mean')
for row in crop_rows[:5]:
    print(f"{row['mask_id']:7d}  {row['area_px']:7d}  "
          f"{row['C1 GFP_mean']:8.1f}  {row['C3 DAPI_mean']:9.1f}")

# %%
rows = read_measurements(TUTORIAL / 'data/nuclear_channel_means.csv')
for plane in ['top', 'middle', 'bottom']:
    print(f"{plane:6s} {sum(r['plane'] == plane for r in rows):6,d} masks")
print(f"total  {len(rows):6,d} masks")

# Verify complete crop measurements against the full middle-plane table.
full_by_id = {r['mask_id']: r for r in rows if r['plane'] == 'middle'}
for row in crop_rows:
    full = full_by_id[row['mask_id']]
    assert row['area_px'] == full['area_px']
    for name in channel_names:
        assert np.isclose(row[name + '_mean'], full[name + '_mean'])
print('All complete crop areas and means match the full-plane table.')

# %%
settings = json.loads((TUTORIAL / 'data/demo_thresholds.json').read_text())
channels = settings['selectedMarkers']
cutoffs = settings['thresholdsRaw']
passing = gate_nuclei(rows, channels, cutoffs)
selected = [row for row, keep in zip(rows, passing) if keep]

print('Middle-plane raw cutoffs:')
for channel in channels:
    print(f"  {channel}: {cutoffs['middle'][channel]:,}")
print(f"Selected: {len(selected):,} / {len(rows):,} "
      f"({100 * len(selected) / len(rows):.1f}%)")

fig = plot_marker_gate(rows, passing, channels, cutoffs)
fig.savefig(RESULTS / 'marker_gate.png', dpi=180)
plt.show()

# %%
fig = plot_spatial_gate(rows, passing, plane='middle')
fig.savefig(RESULTS / 'spatial_gate.png', dpi=180)
plt.show()

middle_selected = sum(keep and row['plane'] == 'middle'
                      for row, keep in zip(rows, passing))
print(f'Middle-plane passing masks: {middle_selected:,}')

# %%
region_export = json.loads((TUTORIAL / 'data/demo_regions.json').read_text())
assert region_export['dataset'] == settings['dataset']
regions = region_export['regions']

fig = plot_spatial_gate(rows, passing, plane='middle', regions=regions)
fig.savefig(RESULTS / 'demo_regions.png', dpi=180)
plt.show()

# %%
summary = count_regions(rows, passing, regions)
print('Region                    Total   Passing   Percent')
for region in summary:
    print(f"{region['region']:25s} {region['total_masks']:5,d}   "
          f"{region['above_threshold_masks']:5,d}    "
          f"{region['above_threshold_percent']:.1f}%")

fig = plot_region_summary(summary)
fig.savefig(RESULTS / 'region_summary.png', dpi=180)
plt.show()

# %%
write_csv(RESULTS / 'gated_nuclei.csv',
          [dict(row, passes_gate=keep) for row, keep in zip(rows, passing)],
          list(rows[0]) + ['passes_gate'])
write_csv(RESULTS / 'selected_nuclei.csv', selected, list(rows[0]))
write_csv(RESULTS / 'region_counts.csv', summary, list(summary[0]))

analysis_settings = dict(
    selected_channels=channels,
    thresholds_raw=cutoffs,
    regions=regions,
    total_masks=len(rows),
    passing_masks=len(selected),
)
(RESULTS / 'analysis_settings.json').write_text(
    json.dumps(analysis_settings, indent=2) + '\n'
)
print('Saved gated_nuclei.csv, selected_nuclei.csv, region_counts.csv')
print('Saved analysis_settings.json')

# %%
import platform
import matplotlib
print('Python:', platform.python_version())
print('NumPy:', np.__version__)
print('tifffile:', tifffile.__version__)
print('Matplotlib:', matplotlib.__version__)
