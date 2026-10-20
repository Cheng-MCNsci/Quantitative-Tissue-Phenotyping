Original UVA HPC source workflows

The two .py files are unmodified Python exports recovered from the user's HPC
workspace. Their SHA-256 hashes are in ../data/original_script_manifest.json.
The *_source.ipynb files preserve the original cell sources and kernel metadata,
with historical outputs and execution counts removed for distribution.

These are interactive development records with diagnostic and tuning trials.
They assume the original input layout and exact example image dimensions.
Run selected notebook cells deliberately; executing an entire historical .py
file also executes the earlier trials and export steps.

Use ../Segmentation_Tutorial.ipynb and ../code/nuclear_segmentation.py for the
short accepted workflow. No model weights or authentication credentials are
included. The untouched notebooks remain in the local HPC_Workflow_Archive.
