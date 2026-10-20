# Quantitative Tissue Phenotyping

Author and pipeline lead: Yang Cheng, University of Virginia (UVA), Christopher Deppmann and Melanie Worley laboratories.

Developed with assistance from ChatGPT (OpenAI) in coding, documentation and tutorial preparation.

[Download the complete code, data and interface bundle](https://github.com/Cheng-MCNsci/Quantitative-Tissue-Phenotyping/releases/download/v0.1.0/Quantitative_Tissue_Phenotyping_Tutorial.zip). Extract it and open `START_HERE.html` for the step-by-step instructions. You can also clone this repository to keep the same folder structure.

This guide tells you which file to open or script to run, in what order, and what each stage reads and writes. Read it in your browser as [START_HERE.html](START_HERE.html), or follow the illustrated [main tutorial](index.html).

## Step 1: extract the download and choose a starting point

Extract `Quantitative_Tissue_Phenotyping_Tutorial.zip` into a new folder. Keep its folders together. Open a terminal in the extracted folder: commands below assume this is your working directory, where `guided_analysis.py`, `data/`, `code/` and `interface/` are visible. Use a fresh extracted copy if you want to preserve a previous run.

1. To try the interface without installing Python, follow Step 2. No script or segmentation run is needed first.
2. To reproduce the included data analysis, follow Steps 3–5. Start with `guided_analysis.py` or `Guided_Tutorial.ipynb`. These use existing example masks, so skip DeepCell.
3. To analyze your own images, follow Step 6. The order is FIJI projections → DeepCell segmentation if needed → raw-intensity measurement → gates and region counts.

The bundle includes a real four-channel crop with 416 complete masks, a full measurement table with 21,480 plane-specific masks, demonstration thresholds and rectangles, and a working interface preloaded with the three wing-disc views. Full projected TIFF files and model weights are not included; full-resolution example image pixels are embedded inside the interface. No DeepCell credentials are included.

## Step 2: explore the example in the browser

Open [interface/index.html](interface/index.html) in a desktop browser such as Chrome. Keep `interface/plotly.min.js` beside it. The interface can run locally without Python, an HPC account or an API key. The HTML is large because it embeds the image pixels.

1. Choose a plane and three marker channels. Set your cutoffs and inspect the scatterplots and the raw/outline image pair. Initially all cutoffs may be zero; that is not the demonstration gate used by the Python tutorial.
2. To reproduce the tutorial's demonstration gate, use Import thresholds and select `data/demo_thresholds.json` from the extracted folder. This selects GFP, RFP and C2, with raw cutoffs 10,000, 10,000 and 5,000. Select All Planes to see the expected combined selection of 5,222 out of 21,480 masks. Import regions with `data/demo_regions.json` to load the two demonstration rectangles.
3. For your own anatomical selection on this example, choose Freehand draw, trace a region on an image, enter its name and select Save region. Repeat for each region. The region table updates with the active gate.
4. Use Export thresholds, Export regions JSON and Export region counts CSV. Keep the JSON exports with the CSV so the selections can be reproduced. Browser storage alone is not a portable backup.

Input: embedded example image pixels and measurements, plus optional imported settings and region JSON. Output: downloaded threshold JSON, region JSON and region-count CSV. Your browser chooses the download location; copy files into the bundle folder if using them in Step 5. Some browser imports replace saved regions or thresholds, so export your current work first if you need to keep it.

This interface does not perform segmentation or import arbitrary new images. Displaying a new dataset currently requires adapting its asset builder and rebuilding the interface. The Python modules can already be used with your own paired images and masks.

## Step 3: prepare Python for the measurement example

Use Python 3.10 or newer. In the extracted folder, create a local environment and install the measurement dependencies:

```bash
python -m venv .venv
```

On macOS or Linux, activate it:

```bash
source .venv/bin/activate
```

On Windows PowerShell, activate it with `.venv\Scripts\Activate.ps1`; in Command Prompt, use `.venv\Scripts\activate.bat`. Then install the included requirements:

```bash
python -m pip install -r requirements.txt
```

Input: `requirements.txt`. Output: an environment with NumPy, tifffile and Matplotlib. These requirements are for measurement and plotting; they do not install DeepCell or reproduce the UVA GPU environment.

## Step 4: run the complete supplied example

Run this script first for the Python example:

```bash
python guided_analysis.py
```

Input: `data/middle_crop_raw.tif`, `data/middle_crop_labels.tif`, `data/nuclear_channel_means.csv`, `data/demo_thresholds.json` and `data/demo_regions.json`. The script imports the reusable modules from `code/` automatically.

The script performs these stages in order: inspect the raw crop and masks → measure 416 complete nuclei → check crop measurements against the full-plane table → gate all three planes → plot the selected population → count demonstration regions → export tables and settings. It uses the saved labels and does not run DeepCell.

Output: `results/crop_measurements.csv`, `results/gated_nuclei.csv`, `results/selected_nuclei.csv`, `results/region_counts.csv` and `results/analysis_settings.json`. It also saves `mask_qc.png`, `marker_gate.png`, `spatial_gate.png`, `demo_regions.png` and `region_summary.png` in `results/`. The script saves plots without opening figure windows. The bundle already contains reference results in this folder; rerunning replaces those same filenames. Copy the folder first if you want to keep them separately.

Expected results: 416 measured crop nuclei; full-plane counts of 6,847 top, 7,253 middle and 7,380 bottom; 5,222 passing masks overall. The middle-plane demonstration rectangles contain 1,334 passing of 3,693 total masks and 312 passing of 1,143 total masks. These rectangles and thresholds are teaching examples, not anatomical ground truth or validated biological positivity criteria.

If you prefer a notebook, use it instead of the script; you do not need to run both:

```bash
python -m pip install jupyterlab
python -m jupyterlab
```

Open `Guided_Tutorial.ipynb`, select the local measurement Python kernel, and run its code cells from top to bottom. Its inputs and output paths match the script, and its plots display inline. GPU instructions shown in Markdown are explanatory; this measurement notebook does not execute segmentation.

## Step 5: run individual scripts and use exported selections

To measure the included raw crop with its existing masks, run:

```bash
python code/measure_nuclei.py --raw data/middle_crop_raw.tif --labels data/middle_crop_labels.tif --channels "C1 GFP" "C2 unknown" "C3 DAPI" "C4 RFP" --plane middle_crop --pixel-size-um 0.5546006944444444 --output results/first_run/crop_measurements.csv
```

Input: matching raw CYX TIFF and label YX TIFF. Output: `results/first_run/crop_measurements.csv`, with one row per positive label, its mask ID, pixel area, physical area, centroid and raw mean intensity for each channel. The output parent folder is created automatically. This command measures existing labels; it does not create them.

Next, reproduce the full demonstration gate and region counts using the supplied full-plane CSV, rather than the crop CSV:

```bash
python code/phenotype_analysis.py --measurements data/nuclear_channel_means.csv --settings data/demo_thresholds.json --regions data/demo_regions.json --output-dir results/first_run/full_example
```

Input: the full-plane measurement CSV and demonstration JSON files. Output: `gated_nuclei.csv`, `selected_nuclei.csv`, `region_counts.csv` and `analysis_settings.json` inside `results/first_run/full_example/`. The last file records settings and input hashes. Expected result: 5,222 / 21,480 passing masks and two region summaries. This script calculates tables, not plot images; use Step 4 for plots.

The crop has its own coordinate origin and uses plane name `middle_crop`. Do not apply full-plane polygons or unmodified full-plane settings to its measurement CSV. Use the full-plane table for the supplied region demonstration.

To reproduce a selection you made in the interface, copy its exported JSON files into a folder named `my_exports/`, rename them as shown below, and run:

```bash
python code/phenotype_analysis.py --measurements data/nuclear_channel_means.csv --settings my_exports/thresholds.json --regions my_exports/regions.json --output-dir results/my_selection
```

Input: the matching example measurement table and your exported threshold/region JSON. Output: the same four analysis files in `results/my_selection/`, reflecting your selection. Threshold and region exports must belong to the same dataset. If no regions were saved, the region CSV has a header and no region rows. Always pair a measurement CSV with exports from the same images and masks; the CSV itself does not encode a dataset identifier.

## Step 6: process your own images in order

### 6.1 Prepare projected raw images in FIJI

Input: your multichannel Z-stack with a suitable nuclear marker. Follow the plane-selection instructions in the main tutorial; the numerical wing-disc Z ranges are example-specific. Preserve channel values and calibration, and export each selected projection as a CYX TIFF. Output: one unenhanced multichannel projection per plane. Record channel order, plane names, pixel size and selected slices.

### 6.2 Create masks in the compatible DeepCell environment

Skip this stage if you already have aligned, reviewed YX label masks. Otherwise move the projected TIFFs and the tutorial's `code/` folder and `Segmentation_Tutorial.ipynb` to your HPC working folder. Select the existing UVA kernel DeepCell ICC GPU (`deepcell_icc_gpu`). See [segmentation_environment.txt](segmentation_environment.txt) for observed versions and its limitations. Do not install the local measurement requirements over the working DeepCell environment.

First run the environment-recording cell in `Segmentation_Tutorial.ipynb`. Input: the active kernel. Output: `deepcell_environment_capture.json`, containing package versions and GPU visibility. This cell does not run inference. Use a new filename or archive the existing capture before rerunning it.

Then edit the notebook's `INPUT`, `OUTPUT`, `files`, `DAPI_CHANNEL` and `pixel_size_um` values. Run its cells in order: load a representative projection → enhance DAPI → load the pretrained model → predict nuclei → apply tissue and boundary filters → visually review QC → save the representative plane. Run the optional remaining-plane loop after reviewing the settings.

Input: your full projected TIFFs and your own model access in the compatible environment. Output: a new folder per plane containing enhanced DAPI, unfiltered predictions, final nuclear labels, a tissue mask, boundary QC JSON and `segmentation_settings.json`. The representative-plane notebook also exports a measurement CSV in its last cell. The batch convenience function saves segmentation outputs; measure those remaining planes in 6.3. Inspect each plane's QC before using its labels. No local GPU rerun is claimed for this notebook.

The reusable `code/nuclear_segmentation.py` combines enhancement, inference, filtering and saving. After settings are accepted, its command-line form is:

```bash
python code/nuclear_segmentation.py --raw my_inputs/middle_4channels.tif --plane middle --dapi-channel 2 --pixel-size-um 0.5546006944444444 --output-dir my_run/middle
```

Replace all example paths, the DAPI channel index and calibration with your own. Input: a raw CYX projection. Output: the segmentation files listed above under `my_run/middle/`. The output directory must be new. Run this command in the DeepCell environment; it loads the model and performs inference. It applies filters automatically, so review the saved outputs before measurement. It does not create the raw-channel measurement CSV.

The files in `source_workflows/` are the original historical scripts and source notebooks. They include tuning trials and assume the original input layout. They are supplied for provenance and reference; use the accepted notebook or reusable module above as the routine entry point.

### 6.3 Measure every plane using raw channels and final masks

After copying reviewed masks back to your analysis computer, activate the measurement environment from Step 3. Run one measurement command for each plane. For example:

```bash
python code/measure_nuclei.py --raw my_inputs/middle_4channels.tif --labels my_run/middle/middle_nuclear_labels.tif --channels "C1 GFP" "C2 unknown" "C3 DAPI" "C4 RFP" --plane middle --pixel-size-um 0.5546006944444444 --output my_run/middle_measurements.csv
```

Input: the original unenhanced CYX projection and its final YX labels on exactly the same pixel grid. Output: the per-nucleus measurement CSV. Replace the channel names and order, plane name and physical pixel size. Never measure the enhanced DAPI core image as a substitute for raw marker channels. Keep plane plus mask ID as the object identifier.

Repeat for top and bottom if applicable. To analyze several planes together, combine their CSVs into one table with one header and consistent channel columns, retaining each row's `plane` value. Do not reuse the supplied wing-disc measurement table for a new specimen.

### 6.4 Define gates and regions, then quantify

Input: your new measurement table, chosen raw cutoffs and any region polygons in the same original-pixel coordinate system. Start from the structure in `data/demo_thresholds.json` and `data/demo_regions.json`, replacing plane names, marker names, cutoffs, region vertices and the dataset identifier with values for your specimen. Keep matching identifiers in both JSON files. The preloaded example interface cannot define regions on a new image until its assets are rebuilt for that specimen.

Run `code/phenotype_analysis.py` as in Step 5, substituting your table and settings. Output: all nuclei with pass flags, selected nuclei, region summaries and an input/settings manifest. This script gates and counts existing measurements; it does not validate segmentation quality or choose biologically meaningful cutoffs for you. Use suitable controls and reviewed anatomical definitions for your analysis.

## File roles and what to keep

1. `guided_analysis.py` or `Guided_Tutorial.ipynb`: complete local worked example; choose one. Reads bundled data, writes analysis tables and figures in `results/`.
2. `code/measure_nuclei.py`: raw image plus existing labels → one measurement row per nucleus. Run after segmentation, or directly with the bundled crop masks.
3. `code/phenotype_analysis.py`: measurement CSV plus threshold/region JSON → gate flags, selected objects, regional counts and settings manifest. Run after measurement, or directly with the bundled full-plane CSV.
4. `code/tutorial_plots.py`: plotting functions imported by the worked example; it is not a separate stage to run.
5. `Segmentation_Tutorial.ipynb` and `code/nuclear_segmentation.py`: optional GPU stage before measurement, for a compatible DeepCell environment. Preprocessing and model/filter settings are included in the outputs.
6. `code/capture_deepcell_environment.py`: optional environment-recording helper; run inside the actual DeepCell kernel. It does not segment images.
7. `interface/index.html`: browser interface for the supplied example; exports settings, regions and counts. It does not create new masks.
8. `source_workflows/`: original development scripts; read its README before running individual historical trial cells.

Keep the unenhanced projections, reviewed final masks, per-object CSVs, threshold and region JSON, segmentation/environment records and exported summaries together for each specimen. The three example projections are independent 2D views; their combined mask total is not a count of unique cells throughout the tissue.
