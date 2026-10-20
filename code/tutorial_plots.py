"""Small plotting helpers for the guided tutorial; all inputs remain explicit."""
import numpy as np
import matplotlib.pyplot as plt


def plot_mask_qc(raw_dapi, labels, pixel_size_um=0.5546006944444444):
    """Show original DAPI and inner label outlines with identical contrast."""
    lo, hi = np.percentile(raw_dapi, [1, 99.8])
    gray = np.clip((raw_dapi.astype(float)-lo)/max(hi-lo,1),0,1)
    p = np.pad(labels,1)
    edge = (labels>0)&((labels!=p[:-2,1:-1])|(labels!=p[2:,1:-1])|
                      (labels!=p[1:-1,:-2])|(labels!=p[1:-1,2:]))
    rgb=np.repeat(gray[...,None],3,axis=2);rgb[edge]=[0,1,1]
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for ax,im,title in zip(axes,[gray,rgb],['Raw C3 DAPI','Retained nuclear outlines']):
        ax.imshow(im,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.set_title(title,loc='left');ax.axis('off')
        h=labels.shape[0];ax.plot([12,12+10/pixel_size_um],[h-16,h-16],color='white',lw=3)
        ax.text(12,h-23,'10 µm',color='white',fontsize=9)
    fig.tight_layout()
    return fig


def plot_marker_gate(rows, passing, selected_channels, thresholds_raw):
    """Show the selected population in three pairwise raw-intensity plots."""
    if len(selected_channels)!=3 or len(passing)!=len(rows):
        raise ValueError('Provide three channels and one flag per row')
    pairs=[(0,1),(0,2),(1,2)]
    colors={'top':'#16828d','middle':'#c18435','bottom':'#8060a5'}
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,(i,j) in zip(axes,pairs):
        x,y=selected_channels[i],selected_channels[j]
        ax.scatter([r[x+'_mean'] for r in rows],[r[y+'_mean'] for r in rows],s=2,color='#d2d6d9',rasterized=True)
        for plane in sorted({r['plane'] for r in rows}):
            selected=[r for r,keep in zip(rows,passing) if keep and r['plane']==plane]
            ax.scatter([r[x+'_mean'] for r in selected],[r[y+'_mean'] for r in selected],s=3,label=plane,color=colors.get(plane),rasterized=True)
        # Separate lines if planes have different thresholds.
        for cut in sorted({thresholds_raw[r['plane']][x] for r in rows}):ax.axvline(cut,color='#526579',ls='--',lw=.8)
        for cut in sorted({thresholds_raw[r['plane']][y] for r in rows}):ax.axhline(cut,color='#526579',ls='--',lw=.8)
        ax.set(xlabel=x+' raw mean',ylabel=y+' raw mean');ax.ticklabel_format(style='sci',axis='both',scilimits=(3,3))
        ax.spines[['top','right']].set_visible(False)
    axes[-1].legend(frameon=False,markerscale=3,fontsize=9)
    fig.tight_layout()
    return fig


def plot_spatial_gate(rows, passing, plane='middle', regions=None):
    """Show all centroids and passing centroids in original image pixel coordinates."""
    fig,ax=plt.subplots(figsize=(9,4.8))
    all_rows=[r for r in rows if r['plane']==plane]
    selected=[r for r,keep in zip(rows,passing) if keep and r['plane']==plane]
    ax.scatter([r['centroid_x'] for r in all_rows],[r['centroid_y'] for r in all_rows],s=2,color='#cbd0d4',label='All retained',rasterized=True)
    ax.scatter([r['centroid_x'] for r in selected],[r['centroid_y'] for r in selected],s=3,color='#007f8c',label='Passes all cutoffs',rasterized=True)
    for region,color in zip(regions or [],['#b77b16','#9c4c88','#32659b']*10):
        if region['plane']!=plane:continue
        points=region['points']+[region['points'][0]]
        ax.plot([p[0] for p in points],[p[1] for p in points],color=color,lw=1.5)
        ax.text(points[0][0],points[0][1]-22,region['name'],color=color,fontsize=9)
    ax.set(xlabel='x (original pixels)',ylabel='y (original pixels)',title=plane.capitalize()+' plane')
    ax.invert_yaxis();ax.set_aspect('equal');ax.legend(frameon=False,markerscale=3,fontsize=9,loc='lower right')
    ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    return fig


def plot_region_summary(summary):
    """Show regional percentages; labels retain the numerator and denominator."""
    shown=[r for r in summary if r['above_threshold_percent'] is not None]
    fig,ax=plt.subplots(figsize=(8,3.4))
    names=[r['region'] for r in shown]; values=[r['above_threshold_percent'] for r in shown]
    bars=ax.barh(names,values,color=['#b77b16','#9c4c88']*len(shown))
    for bar,r in zip(bars,shown):
        ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,
                f"{r['above_threshold_percent']:.1f}%  ({r['above_threshold_masks']:,} / {r['total_masks']:,})",va='center',fontsize=10)
    ax.set(xlabel='Nuclei passing the gate (%)',xlim=(0,max([50]+values)+18))
    ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    return fig
