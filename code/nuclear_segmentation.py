"""Accepted wing-disc segmentation workflow recovered from UVA HPC.

DeepCell inference belongs in a compatible DeepCell environment. Importing this
module does not load TensorFlow, contact DeepCell, or request an API key.
"""
from pathlib import Path
import argparse
import copy
import json
import numpy as np
import tifffile

ENHANCEMENT = dict(smoothing_sigma_px=0.5, background_sigma_px=12.0, core_sigma_px=1.8)
SETTINGS = dict(scale_factor=3, tissue_smoothing_sigma_px=12.0,
    tissue_threshold_multiplier=0.5, tissue_closing_radius_px=32,
    tissue_min_component_area_px=5000, minimum_tissue_overlap=0.5,
    boundary_band_px=10, ring_inner_px=2, ring_outer_px=5,
    minimum_local_ratio=1.2, pixel_size_um=0.5546006944444444,
    model_mpp=0.65, postprocessing=dict(radius=10, maxima_threshold=0.1,
    interior_threshold=0.01, exclude_border=False, small_objects_threshold=0, min_distance=10))


def _raw_image(image):
    raw=np.asarray(image,dtype=np.float32)
    if raw.ndim!=2 or not np.isfinite(raw).all() or np.any(raw<0):
        raise ValueError('Expected finite nonnegative 2D raw DAPI')
    return raw


def enhance_dapi(raw_dapi, smoothing_sigma_px=0.5, background_sigma_px=12.0, core_sigma_px=1.8):
    """Compute the nonnegative scale-normalized negative LoG core response."""
    from scipy.ndimage import gaussian_filter, gaussian_laplace
    raw=_raw_image(raw_dapi)
    if min(smoothing_sigma_px,background_sigma_px,core_sigma_px)<=0:
        raise ValueError('Gaussian scales must be positive')
    smooth=gaussian_filter(raw,smoothing_sigma_px,mode='reflect')
    corrected=np.maximum(smooth-gaussian_filter(smooth,background_sigma_px,mode='reflect'),0)
    return np.maximum(-gaussian_laplace(corrected,core_sigma_px,mode='reflect')*core_sigma_px**2,0).astype(np.float32)


def load_nuclear_model():
    """Load the exact version used on HPC; model download may need an API key.

    Configure authentication in your own environment; no key is stored here.
    """
    from deepcell.applications import NuclearSegmentation
    return NuclearSegmentation.from_version('1.1')


def predict_nuclei(core, app, settings=None):
    """Apply the accepted 3x apparent-size trial and map labels back to YX.

    image_mpp=app.model_mpp deliberately preserves the apparent enlargement.
    This empirical convention does not change physical output calibration.
    """
    from skimage.transform import resize
    cfg=copy.deepcopy(SETTINGS if settings is None else settings)
    core=_raw_image(core)
    scale=cfg['scale_factor']
    if int(scale)!=scale or scale<1:raise ValueError('Scale factor must be a positive integer')
    if not np.isclose(float(app.model_mpp),cfg['model_mpp']):
        raise ValueError('Loaded model calibration differs from the recorded model')
    enlarged=resize(core,tuple(int(scale)*n for n in core.shape),order=1,
                    preserve_range=True,anti_aliasing=False).astype(np.float32)
    prediction=app.predict(enlarged[None,...,None],batch_size=4,
        image_mpp=app.model_mpp,postprocess_kwargs=dict(cfg['postprocessing']))
    if prediction.shape!=(1,*enlarged.shape,1):raise ValueError('Unexpected model output shape')
    return resize(prediction[0,...,0],core.shape,order=0,
                  preserve_range=True,anti_aliasing=False).astype(np.uint32)


def make_tissue_mask(raw_dapi, settings=None):
    """Reproduce tissue smoothing, thresholding, closing, component filter and fill."""
    from scipy.ndimage import gaussian_filter, binary_fill_holes, label
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, disk
    cfg=SETTINGS if settings is None else settings
    signal=gaussian_filter(_raw_image(raw_dapi),cfg['tissue_smoothing_sigma_px'])
    threshold=cfg['tissue_threshold_multiplier']*float(threshold_otsu(signal))
    candidate=binary_closing(signal>threshold,footprint=disk(cfg['tissue_closing_radius_px']))
    # Same 4-connected boolean component rule as the original remove_small_objects.
    components,_=label(candidate)
    counts=np.bincount(components.ravel())
    keep=counts>=cfg['tissue_min_component_area_px'];keep[0]=False
    return binary_fill_holes(keep[components]),threshold


def filter_nuclei(raw_dapi, original_labels, settings=None):
    """Filter existing labels by tissue overlap and local boundary contrast.

    Returns final labels, tissue mask, summary and per-boundary-label QC records.
    Original labels are preserved, and inputs are not modified.
    """
    from scipy.ndimage import distance_transform_edt, binary_dilation
    from skimage.morphology import disk
    from skimage.measure import regionprops
    from skimage.segmentation import find_boundaries
    cfg=SETTINGS if settings is None else settings
    raw=_raw_image(raw_dapi);labels=np.asarray(original_labels)
    if labels.shape!=raw.shape or not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0):
        raise ValueError('Labels must be nonnegative integers with the raw DAPI shape')
    labels=labels.astype(np.uint32,copy=False)
    tissue,threshold=make_tissue_mask(raw,cfg)
    flat=labels.ravel().astype(np.int64);totals=np.bincount(flat)
    inside=np.bincount(flat[tissue.ravel()],minlength=len(totals))
    overlaps=np.divide(inside,totals,out=np.zeros(len(totals)),where=totals>0)
    keep=(totals>0)&(overlaps>=cfg['minimum_tissue_overlap']);keep[0]=False
    tissue_labels=np.where(keep[labels],labels,0).astype(np.uint32)
    edge=find_boundaries(tissue,mode='thick')
    band=distance_transform_edt(~edge)<=cfg['boundary_band_px']
    boundary_ids=set(np.unique(tissue_labels[band]).tolist())-{0}
    records=[]
    for region in regionprops(tissue_labels):
        if region.label not in boundary_ids:continue
        y0,x0,y1,x1=region.bbox;r=cfg['ring_outer_px']
        ya,xa=max(0,y0-r),max(0,x0-r)
        yb,xb=min(raw.shape[0],y1+r),min(raw.shape[1],x1+r)
        nucleus=tissue_labels[ya:yb,xa:xb]==region.label
        ring=(binary_dilation(nucleus,structure=disk(r))&
              ~binary_dilation(nucleus,structure=disk(cfg['ring_inner_px']))&
              (labels[ya:yb,xa:xb]==0))
        evaluable=np.count_nonzero(ring)>=10;ratio=None
        if evaluable:
            patch=raw[ya:yb,xa:xb]
            ratio=float(np.median(patch[nucleus]))/max(float(np.median(patch[ring])),1.0)
        records.append(dict(label=int(region.label),ratio=ratio,evaluable=bool(evaluable),
                            flagged=bool(evaluable and ratio<cfg['minimum_local_ratio'])))
    excluded=[r['label'] for r in records if r['flagged']]
    remove=np.zeros(len(totals),dtype=bool);remove[excluded]=True
    final=tissue_labels.copy();final[remove[final]]=0
    summary=dict(initial_objects=int(np.count_nonzero(totals[1:])),after_tissue=int(np.count_nonzero(keep)),
        boundary_excluded=len(excluded),final_nuclei=int(np.count_nonzero(np.unique(final))),
        tissue_coverage=float(np.mean(tissue)),tissue_threshold=threshold)
    return final,tissue,summary,records


def save_segmentation(output_dir, plane, raw_dapi, app, settings=None, enhancement=None):
    """Run one plane and save raw predictions, final masks, QC and parameters.

    The output directory must be new, preventing silent replacement of a run.
    """
    cfg=copy.deepcopy(SETTINGS if settings is None else settings)
    enhancement=copy.deepcopy(ENHANCEMENT if enhancement is None else enhancement)
    out=Path(output_dir)
    if not plane or Path(plane).name!=plane:raise ValueError('Plane must be a simple name')
    if out.exists():raise FileExistsError(f'Choose a new output folder: {out}')
    raw=_raw_image(raw_dapi)
    core=enhance_dapi(raw,**enhancement)
    initial=predict_nuclei(core,app,cfg)
    final,tissue,summary,qc=filter_nuclei(raw,initial,cfg)
    out.mkdir(parents=True,exist_ok=False)
    cal=cfg['pixel_size_um']
    for name,arr in [('DAPI_core_enhanced',core),('unfiltered_labels',initial),
                     ('nuclear_labels',final),('tissue_mask',tissue.astype(np.uint8))]:
        dest=out/f'{plane}_{name}.tif'
        tifffile.imwrite(dest,arr,photometric='minisblack',metadata={'axes':'YX','unit':'micron'},
                         resolution=(1/cal,1/cal),resolutionunit='NONE')
        if not np.array_equal(tifffile.imread(dest),arr):raise RuntimeError('TIFF read-back verification failed')
    (out/f'{plane}_boundary_qc.json').write_text(json.dumps(qc,indent=2)+'\n')
    manifest=dict(model_class='NuclearSegmentation',model_version='1.1',settings=cfg,
                  enhancement=enhancement,summary=summary,nuclear_input='core-enhanced DAPI')
    (out/'segmentation_settings.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return final,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--plane',required=True)
    p.add_argument('--dapi-channel',type=int,default=2,help='Zero-based channel index')
    p.add_argument('--pixel-size-um',type=float,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.pixel_size_um<=0 or not np.isfinite(a.pixel_size_um):p.error('Pixel size must be positive and finite')
    if a.output_dir.exists():p.error('Choose a new output folder')
    with tifffile.TiffFile(a.raw) as tf:
        if tf.series[0].axes!='CYX':p.error('Expected a projected CYX TIFF')
        raw=tf.asarray()
    if not 0<=a.dapi_channel<raw.shape[0]:p.error('DAPI channel index is out of range')
    cfg=copy.deepcopy(SETTINGS);cfg['pixel_size_um']=a.pixel_size_um
    app=load_nuclear_model()
    _,manifest=save_segmentation(a.output_dir,a.plane,raw[a.dapi_channel],app,cfg)
    print(json.dumps(manifest['summary'],indent=2))


if __name__=='__main__':main()
