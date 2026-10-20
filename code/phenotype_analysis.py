"""Apply raw-intensity gates and count nuclei inside exported freehand regions.

This module uses only Python's standard library. Import its functions or run
`python phenotype_analysis.py --help` for the command-line interface.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math


def read_measurements(path):
    """Read a measurement CSV; preserve IDs and convert measurements to numbers."""
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    seen = set()
    for row in rows:
        for key in ['plane', 'mask_id', 'centroid_x', 'centroid_y']:
            if key not in row:
                raise ValueError(f'Missing CSV column: {key}')
        if not row['plane']:
            raise ValueError('Plane names must not be empty')
        row['mask_id'] = int(row['mask_id'])
        if row['mask_id'] <= 0:
            raise ValueError('Mask IDs must be positive')
        identity = row['plane'], row['mask_id']
        if identity in seen:
            raise ValueError(f'Duplicate plane and mask ID: {identity}')
        seen.add(identity)
        for key in list(row):
            if key in ['centroid_x','centroid_y','area_px','area_um2'] or key.endswith('_mean'):
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise ValueError(f'Nonfinite measurement: {identity}, {key}')
    return rows


def gate_nuclei(rows, selected_channels, thresholds_raw):
    """Return booleans aligned with rows; all selected channels must pass >=.

    thresholds_raw maps plane -> channel name -> cutoff in original raw units.
    Channels may include any nonempty distinct subset of measured channels.
    Example: {'middle': {'C1 GFP': 10000, 'C4 RFP': 10000}}.
    """
    if not selected_channels or len(set(selected_channels)) != len(selected_channels):
        raise ValueError('Select at least one distinct channel')
    passing = []
    for row in rows:
        decisions = []
        for channel in selected_channels:
            try:
                cutoff = float(thresholds_raw[row['plane']][channel])
                value = float(row[channel + '_mean'])
            except KeyError as exc:
                raise ValueError(f'Missing channel or raw threshold: {row["plane"]}, {channel}') from exc
            if not math.isfinite(cutoff) or cutoff < 0 or not math.isfinite(value):
                raise ValueError('Use finite intensities and finite nonnegative raw cutoffs')
            decisions.append(value >= cutoff)
        passing.append(all(decisions))
    return passing


def contains_centroid(x, y, points):
    """Ray-crossing polygon membership, including points on the polygon edge.

    Coordinates are original zero-based image pixels, x then y, as exported by
    the explorer. Edge tolerance matches its JavaScript implementation.
    """
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        cross = (x-xi)*(yj-yi) - (y-yi)*(xj-xi)
        if abs(cross) < 1e-7 and min(xi,xj) <= x <= max(xi,xj) and min(yi,yj) <= y <= max(yi,yj):
            return True
        if (yi > y) != (yj > y) and x < (xj-xi)*(y-yi)/(yj-yi)+xi:
            inside = not inside
        j = i
    return inside


def count_regions(rows, passing, regions):
    """Return total, passing and percent per polygon; overlaps count independently.

    Empty regions have percent=None rather than implying a measured zero rate.
    """
    if len(rows) != len(passing):
        raise ValueError('Provide one gate result per measurement row')
    planes = {r['plane'] for r in rows}
    result = []
    ids = set()
    for region in regions:
        if region['id'] in ids:
            raise ValueError('Region IDs must be unique')
        ids.add(region['id'])
        if region['plane'] not in planes:
            raise ValueError(f'Region plane is absent from measurements: {region["plane"]}')
        if not isinstance(region['name'], str) or not region['name'].strip():
            raise ValueError('Each region needs a name')
        points = region['points']
        if len(points) < 3 or any(len(p) != 2 or not all(math.isfinite(v) and v >= 0 for v in p) for p in points):
            raise ValueError('Each region needs at least three finite nonnegative XY points')
        total = passed = 0
        for row, keep in zip(rows, passing):
            if row['plane'] == region['plane'] and contains_centroid(row['centroid_x'],row['centroid_y'],points):
                total += 1
                passed += bool(keep)
        result.append(dict(plane=region['plane'],region_id=region['id'],region=region['name'],
                           total_masks=total,above_threshold_masks=passed,
                           above_threshold_percent=100*passed/total if total else None))
    return result


def write_csv(path, rows, fieldnames):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurements', type=Path, required=True)
    p.add_argument('--settings', type=Path, required=True, help='Export thresholds JSON, using selectedMarkers and thresholdsRaw')
    p.add_argument('--regions', type=Path, help='Optional Export regions JSON; otherwise use regions embedded in settings')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    settings = json.loads(a.settings.read_text())
    if settings.get('version') != 1:
        raise ValueError('Expected settings version 1')
    rows = read_measurements(a.measurements)
    if not rows:
        raise ValueError('Measurement CSV is empty')
    selected = settings['selectedMarkers']
    cutoffs = settings['thresholdsRaw']
    regions = settings.get('regions', [])
    if a.regions:
        exported = json.loads(a.regions.read_text())
        if exported.get('version') != 1 or exported.get('dataset') != settings.get('dataset'):
            raise ValueError('Region and threshold exports must have the same dataset identifier and version 1')
        regions = exported['regions']
    passing = gate_nuclei(rows,selected,cutoffs)
    counts = count_regions(rows,passing,regions)
    for count in counts:
        for i,ch in enumerate(selected,1):
            count[f'channel_{i}'] = ch
            count[f'raw_cutoff_{i}'] = cutoffs[count['plane']][ch]
    a.output_dir.mkdir(parents=True,exist_ok=True)
    fields = list(rows[0])
    write_csv(a.output_dir/'gated_nuclei.csv',[dict(r,passes_gate=b) for r,b in zip(rows,passing)],fields+['passes_gate'])
    write_csv(a.output_dir/'selected_nuclei.csv',[r for r,b in zip(rows,passing) if b],fields)
    count_fields = ['plane','region_id','region','total_masks','above_threshold_masks','above_threshold_percent']
    for i in range(1,len(selected)+1): count_fields += [f'channel_{i}',f'raw_cutoff_{i}']
    write_csv(a.output_dir/'region_counts.csv',counts,count_fields)
    inputs = [a.measurements,a.settings]+([a.regions] if a.regions else [])
    manifest = dict(total_masks=len(rows),passing_masks=sum(passing),
                    selected_channels=selected,thresholds_raw=cutoffs,regions=regions,
                    settings_dataset=settings.get('dataset'),
                    measurement_dataset_note='CSV has no dataset identifier; user must pair it with its matching exports.',
                    inputs=[dict(file=v.name,sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for v in inputs])
    (a.output_dir/'analysis_settings.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'{sum(passing):,} / {len(rows):,} nuclear masks pass. Wrote {len(counts)} region summaries to {a.output_dir}')


if __name__ == '__main__':
    main()
