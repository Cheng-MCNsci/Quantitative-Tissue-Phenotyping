Quantitative Tissue Phenotyping — example explorer

Open index.html in a desktop browser such as Chrome. Keep plotly.min.js beside it.
No Python, server, HPC account or DeepCell API key is needed to explore this saved
example. The page embeds full-resolution image pixels and measured nuclei.

Choose a plane and three markers; set thresholds; inspect the paired raw and
outline views. Choose Freehand draw, trace a region, name it, and Save region.
Export thresholds JSON, regions JSON and region counts CSV for reproducibility.
Exported settings can be used with the Python tutorial in the parent folder.
Settings saved in browser storage may not follow the page when it is moved.

This version is preloaded with the three wing-disc projections. It does not run
segmentation or load arbitrary new TIFFs. New datasets require adapting the asset
builder and rebuilding the interface. It is an example viewer, not a general
spatial-omics importer. Nuclear mean intensity is not an HCR transcript count.

The HTML is large because it embeds the actual raw image pixels and measurements.
