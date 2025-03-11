import numpy as np
import random
import os, torch
from scipy.special import jv
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from sklearn.metrics import mean_absolute_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

"""Four channel imaging model with OTF"""
def ForwardOTF(gt, otf):
    # add otf to image
    f_img = fft2(gt[:,:])
    f_img = f_img*otf
    imgf = np.real(ifft2(f_img))
    imgf = np.expand_dims(imgf,axis=0)

    return imgf

def Cal_Otf(dx:float=512, rand=True, nr=1.0, ch = "A") -> np.ndarray:
    # 4 channel params
    # scales = {'A':1.12, 'C': 0.897, 'G': 0.92, 'T': 1.069}   #SR
    scales = {'A':2.80, 'T':2.57, 'G':2.29, 'C':2.16}  ## T10 scale
    # scales = {'H':1.55, 'L': 1.29 }   #SR

    #
    dx = int(dx)
    x = np.linspace(0,dx-1,dx)
    y = np.linspace(0,dx-1,dx)
    xx,yy = np.meshgrid(x,y)
    r = np.sqrt(np.square(np.minimum(xx,np.abs(xx-dx)))+np.square(np.minimum(yy,np.abs(yy-dx))))
    eps = 1e-10

    if rand:
        # nr = np.random.rand()*0.2+0.9   # [0.9,1.1]
        # div = 1/(scales[ch]*nr*r+eps)
        # bj = jv(1,scales[ch]*nr*r+eps)
        rand_error = np.random.normal(0, 0.03, 1)[0]
        scale = scales[ch] + rand_error
        div = 1 / (scale * r + eps)
        bj = jv(1, scale * r + eps)
    else:
        # nr = 1
        div = 1/( scales[ch]*nr*r+eps )
        bj = jv( 1,scales[ch]*nr*r+eps )
    psf = np.square( np.abs(2*bj)*div )
    # otf = np.abs(np.fft.fft2(psf))
    otf = np.fft.fft2( psf )
    otf = otf/np.max( np.abs(otf) )

    return otf

# def PsfOtf(dx:int=512, ch = "C") -> (np.ndarray, np.ndarray) :
#     # AIM: To generate PSF and OTF using Bessel function
#     # INPUT VARIABLES
#     #   w: image size
#     #   scale: a parameter used to adjust PSF/OTF width
#     # OUTPUT VRAIBLES
#     #   yyo: system PSF
#     #   OTF2dc: system OTF
#     eps = np.finfo(np.float64).eps
#     scales = {'A': 1.12, 'C': 0.897, 'G': 0.92, 'T': 1.069}  # SR
#     scale = scales[ch]
#
#     x = np.linspace(0, dx-1, dx)
#     y = np.linspace(0, dx-1, dx)
#     X, Y = np.meshgrid(x, y)
#
#     # Generation of the PSF with Besselj.
#     R = np.sqrt(np.minimum(X, np.abs(X-dx))**2+np.minimum(Y, np.abs(Y-dx))**2)
#     yy = np.abs(2*jv(1, scale*R+eps) / (scale*R+eps))**2
#     psf = fftshift(yy)
#
#     # Generate 2D OTF.
#     OTF2d = fft2(yy)
#     OTF2dmax = np.max([np.abs(OTF2d)])
#     OTF2d = OTF2d/OTF2dmax
#     OTF2dc = np.abs(fftshift(OTF2d))
#
#     return (psf, OTF2dc)
def Cal_SIM(dx:int=512, ch:str='H'  ) -> np.ndarray:
    # ModFacs = {'A':0.7, 'C': 0.6, 'G': 0.6, 'T': 0.65 }
    ModFacs = { 'H': 0.6, 'L': 0.6 }
    mA = 1  # mean illumination  intensity
    wo = dx / 2
    # random a phase
    psA = random.uniform(0, np.pi)

    # Illumination patterns
    thetaA = -105 * np.pi / 180
    k2 = 162.7

    k2a = (k2 / dx) * np.array( [np.cos(thetaA), np.sin(thetaA)] )

    # grid generation
    x = np.linspace(0, dx - 1, dx)
    y = np.linspace(0, dx - 1, dx)
    xx, yy = np.meshgrid(x, y)

    # amplitude of illumination intensity above mean
    aA = 1 * ModFacs[ch]
    # illunination patterns
    sAo = mA + aA * np.cos(2 * np.pi * (k2a[0] * (xx - wo) + k2a[1] * (yy - wo)) + psA)
    return sAo

def Add_SIM(gt, sim, otf):
    """
    Geneerate raw wim images
    input:
        GT: synthetic HR image
        sim: simulate sim
        otf: optical transfer funtion
    """
    # n_ch = gt.shape[0]
    # chs = ["A.", "C.", "G.", "T."]
    # for ch in range(n_ch):
    # f_img = np.fft.fft2(gt)
    f_img = gt * sim
    imgf = np.real(np.fft.ifft2( fft2(f_img)*otf ) )
    imgf = np.expand_dims(imgf, axis=0)

    return imgf

def evaluation_metrics(gt_im, pred_im):
    """
        calculate the evaluation metrics between gt and pred image
        return mae, psnr, ssim
    """
    gt_im = (gt_im - gt_im.min() ) / (gt_im.max() - gt_im.min())*255
    pred_im = (pred_im - pred_im.min()) / (pred_im.max() - pred_im.min()) * 255

    mae = mean_absolute_error(gt_im, pred_im)
    psnr = peak_signal_noise_ratio(gt_im, pred_im, data_range=255)
    ssim = structural_similarity(gt_im, pred_im, data_range=255)

    return mae, psnr, ssim


DEFAULT_RANDOM_SEED = 2023


def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# basic + tensorflow + torch
def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)