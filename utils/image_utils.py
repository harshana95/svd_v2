import shutil
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import einops
import os
import zipfile
from PIL import Image

def laplacian_highpass_filter(img):
    dtype = img.dtype
    device = img.device
    img = np.copy(img.numpy()*255).astype(np.uint8)
    img = einops.rearrange(img, 'c h w -> h w c')
    is_gray = img.shape[-1] == 1
    # Apply Gaussian blur to the image to reduce noise
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    # # Laplacian kernel for high-pass filtering
    # laplacian_kernel = np.array([[0, 0, 1, 1, 1, 0, 0],
    #                              [0, 1, 1, 2, 1, 1, 0],
    #                              [1, 1, 2, -8, 2, 1, 1],
    #                              [1, 2, -8, -24, -8, 2, 1],
    #                              [1, 1, 2, -8, 2, 1, 1],
    #                              [0, 1, 1, 2, 1, 1, 0],
    #                              [0, 0, 1, 1, 1, 0, 0]])
    
    # laplacian_kernel = np.array([[0, 1, 0],
    #                              [1, -4, 1],
    #                              [0, 1, 0]])
    # # Apply the Laplacian filter
    # lf = cv2.filter2D(img, -1, laplacian_kernel)
    
    lf = cv2.Laplacian(img,-1, ksize=1)
#     lf = ((lf - lf.min())/(lf.max() - lf.min())*255).astype(np.uint8)
    if is_gray:
        lf = lf[:,:, None]
    highpass_image = einops.rearrange(lf, 'h w c -> c h w')
    return torch.from_numpy(highpass_image).to(dtype).to(device)/255.0

def laplacian_highpass_filter_guo(img, downOrder=2, **kwargs): # img is a tensor
    dtype = img.dtype
    device = img.device
    img = img.numpy()
    is_grayscale = img.shape[0] == 1  
    img = np.copy(img)
    img = (einops.rearrange(img, 'c h w -> h w c')*255).astype(np.uint8)
    I_lowres = np.copy(img)
    h,w,c = I_lowres.shape
    for i in range(c):
        tmp = I_lowres[:,:,i]
        h,w = tmp.shape
        for _ in range(downOrder):
            h = h//2
            w = w//2
            tmp = cv2.pyrDown(tmp,dstsize=(h, w))
        for _ in range(downOrder):
            h *= 2
            w *= 2
            tmp = cv2.pyrUp(tmp,dstsize=(h, w))
        I_lowres[:,:,i] = tmp
        
    I_edge = np.uint8(np.abs(np.float32(img)-np.float32(I_lowres)))
    I_edge = ((I_edge - I_edge.min())/(I_edge.max() - I_edge.min())*255).astype(np.uint8)
    
    # if is_grayscale:
    #     I_edge = cv2.fastNlMeansDenoising(I_edge,None,15,5,15)
    #     # fastNlMeansDenoising will drop the last dim if grayscale
    #     I_edge = I_edge[:, :, None]    
    # else:
    #     I_edge = cv2.fastNlMeansDenoisingColored(I_edge,None,15,5,15)

    I_edge = einops.rearrange(I_edge, 'h w c -> c h w')
    
    out = torch.from_numpy(I_edge).to(dtype).to(device)/255.0  # cv2 handles images in 0-255, rescale to 0-1
    return out



def save_images_as_zip(images, root_path, zip_filename="images.zip"):
    folder = os.path.join(root_path, 'images', 'tmp')
    os.makedirs(folder, exist_ok=True)
    
    # Remove existing files inside the folder
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    
    # saving png files
    image_paths = []
    for i, img_array in enumerate(images):
        img = einops.rearrange(img_array, 'c h w -> h w c')
        img_path = os.path.join(folder, f"image_{i+1}.png")
        plt.imsave(img_path, img)
        image_paths.append(img_path)
    
    with zipfile.ZipFile(os.path.join(root_path, 'images', zip_filename), 'w', zipfile.ZIP_DEFLATED) as zipf:
        for img_path in image_paths:
            zipf.write(img_path, os.path.basename(img_path))


def display_images(d, same_plot=None, independent_colorbar=False, row_limit=-1, cols_per_plot=1, size=1):
    im = None
    if same_plot is None:
        same_plot = []
    max_rows = 0
    vmin = 1e10
    vmax = -1e10
    # find vmax, vmin, max rows
    for k in d.keys():
        # if max_rows > 0:
        #     assert max_rows == len(d[k]), "Lengths of each array should be same"
        max_rows = max(max_rows, len(d[k]))
        if max_rows == 0:
            print("Cannot display images! Zero sized array. Skipping...")
            return
        if len(d[k].shape) == 3:
            vmin = min(vmin, d[k].min())
            vmax = max(vmax, d[k].max())
    if independent_colorbar:
        vmin, vmax = None, None

    # set key map. if keys should be in the same plot, they will be grouped by a list
    keys = [key for key in d.keys()]
    plotmap = [same for same in same_plot]
    for same in same_plot:
        for key in same:
            keys.remove(key)
    for key in keys:
        plotmap.append([key])

    # start plotting
    rows, cols = max_rows, len(plotmap)
    cols *= cols_per_plot
    rows = int(np.ceil(rows / cols_per_plot))
    if row_limit > 0:
        rows = min(rows, row_limit)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * size, rows * size))
    show_legend = False
    for i in range(rows * cols):
        x, y = i // cols, i % cols
        if rows * cols == 1:
            ax = axes
        else:
            ax = axes.flat[i]
        for key in plotmap[y // cols_per_plot]:
            if x * cols_per_plot + (y % cols_per_plot) >= len(d[key]):
                ax.axis(False)
                continue
            data = d[key][x * cols_per_plot + (y % cols_per_plot)]
            if len(data.shape) == 2:
                im = ax.imshow(data, vmin=vmin, vmax=vmax)
                ax.axis(False)
            elif len(data.shape) == 3:
                im = ax.imshow(data)
                ax.axis(False)
            else:
                ax.plot(data[:], label=key)
                show_legend = True
            if im is not None and independent_colorbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')

            if x == 0:
                ax.set_title(f"{key}")
            if y == 0:
                ax.set_ylabel(f"{x}")
        if x == 0 and len(data.shape) != 2 and show_legend:
            ax.legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if im is not None and not independent_colorbar:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.35, 0.01, 0.4])
        fig.colorbar(im, cax=cbar_ax)
    return fig