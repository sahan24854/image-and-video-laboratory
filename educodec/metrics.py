"""Full-reference RGB error metrics and Gaussian-window luminance SSIM."""
import numpy as np
from scipy.ndimage import gaussian_filter


def quality(original, reconstructed):
    a=np.asarray(original,dtype=float); b=np.asarray(reconstructed,dtype=float)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[-1]!=3: raise ValueError('Expected matching RGB images')
    error=a-b; mse=float(np.mean(error**2))
    weights=np.array([.299,.587,.114]); x=a@weights; y=b@weights
    smooth=lambda v:gaussian_filter(v,1.5,truncate=3.5,mode='reflect')
    ux,uy=smooth(x),smooth(y)
    vx=np.maximum(smooth(x*x)-ux*ux,0); vy=np.maximum(smooth(y*y)-uy*uy,0)
    cov=smooth(x*y)-ux*uy
    ssim=((2*ux*uy+2.55**2)*(2*cov+7.65**2)/((ux*ux+uy*uy+2.55**2)*(vx+vy+7.65**2))).mean()
    return dict(psnr_db=float(10*np.log10(255**2/mse)) if mse else float('inf'),ssim_y=float(ssim),mse=mse,rmse=float(np.sqrt(mse)),mae=float(np.mean(np.abs(error))),max_error=float(np.max(np.abs(error))))
