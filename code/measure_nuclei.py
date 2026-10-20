"""Measure raw channel pixels within existing 2D nuclear labels."""
from pathlib import Path
import argparse
import csv
import numpy as np
import tifffile


def measure_nuclei(raw, labels, channel_names, pixel_size_um=None):
    """Return one record per positive label; raw must have CYX axes.

    Centroids are zero-based pixel coordinates. Labels need not be consecutive.
    Background is label 0. No filtering or background subtraction is applied.
    """
    raw, labels = np.asarray(raw), np.asarray(labels)
    if raw.ndim != 3 or labels.ndim != 2 or raw.shape[1:] != labels.shape:
        raise ValueError('Expected raw CYX and labels YX with matching XY shape')
    if len(channel_names) != raw.shape[0] or len(set(channel_names)) != len(channel_names):
        raise ValueError('Provide a unique name for each channel')
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError('Labels must be nonnegative integers')
    if not np.isfinite(raw).all():
        raise ValueError('Raw intensities must be finite')
    if pixel_size_um is not None and (not np.isfinite(pixel_size_um) or pixel_size_um <= 0):
        raise ValueError('Pixel size must be positive and finite')
    ids, inverse = np.unique(labels, return_inverse=True)
    inverse = inverse.ravel()
    counts = np.bincount(inverse)
    yy, xx = np.indices(labels.shape)
    cx = np.bincount(inverse, weights=xx.ravel()) / counts
    cy = np.bincount(inverse, weights=yy.ravel()) / counts
    means = [np.bincount(inverse, weights=c.ravel()) / counts for c in raw]
    records = []
    for i in np.flatnonzero(ids > 0):
        row = dict(mask_id=int(ids[i]), area_px=int(counts[i]),
                   centroid_x=float(cx[i]), centroid_y=float(cy[i]))
        if pixel_size_um is not None:
            row['area_um2'] = float(counts[i] * pixel_size_um ** 2)
        row.update({name + '_mean': float(means[c][i]) for c, name in enumerate(channel_names)})
        records.append(row)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--channels', nargs='+', required=True)
    p.add_argument('--plane', required=True)
    p.add_argument('--pixel-size-um', type=float)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    with tifffile.TiffFile(a.raw) as tif:
        if tif.series[0].axes != 'CYX':
            raise ValueError('Input must explicitly have CYX axes; reorder other inputs first')
        raw = tif.asarray()
    rows = measure_nuclei(raw, tifffile.imread(a.labels), a.channels, a.pixel_size_um)
    if not rows:
        raise ValueError('No positive nuclear labels found')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['plane'] + list(rows[0]))
        w.writeheader()
        w.writerows(dict(plane=a.plane, **r) for r in rows)
    print(f'Wrote {len(rows)} nuclei to {a.output}')


if __name__ == '__main__':
    main()
