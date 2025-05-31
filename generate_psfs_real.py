import glob
import os
import shutil
import time
import cv2
import einops
import matplotlib
from matplotlib import patches
import scipy

from utils.dataset_utils import crop_arr

matplotlib.use("TkAgg")
import argparse
import numpy as np
from matplotlib import pyplot as plt
from os.path import join

from psf.svpsf import PSFSimulator
from utils import generate_folder

# from PySide2.QtCore import QLibraryInfo
# from PyQt5.QtCore import QLibraryInfo
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

boxes = []

drawing, ix, iy, cx, cy = False, 0, 0, 0, 0
fac = 1
psf = None
to_view = None

metadata = {'center':[], 'position_index':[], 'index':[], 'psf_metadata':[]}
psfs = {}

def extract_filename(fname):
    filename = os.path.basename(fname)
    for i in range(len(filename) - 1, -1, -1):
        if filename[i] == '.':
            filename = filename[:i]
            break
    split = filename.split(',')
    if len(split) == 2:
        return {'x': int(split[0]), 'y': int(split[1])}
    elif len(split) == 3:
        return {'x': int(split[0]), 'y': int(split[1]), 'exposure': float(split[2])}
    else:
        return {'x': -1, 'y': -1}

def estimate_box(_psf, n=5, plot=False, return_data=False):
    box = _estimate_box(_psf, n, plot, False)
    #reestimate using the crop
    xs, xe, ys, ye = box[-2][1], box[-1][1], box[-2][0], box[-1][0]
    ps, pe, qs, qe = max(0, xs), min(xe, _psf.shape[0]), max(0, ys), min(ye, _psf.shape[1])
    cropped_psf = _psf[ps:pe, qs:qe]
    box2 = _estimate_box(cropped_psf, n, plot, False)
    return [[qs+box2[0][0],ps+box2[0][1]], [qs+box2[1][0], ps+box2[1][1]]]

def _estimate_box(_psf, n=5, plot=False, return_data=False):
    """
    @param _psf:
    @param n:
    @return: [[x1,y1], [x2,y2]] 1-top left 2-bottom right
    """
    _psf = _psf.copy().astype(float).mean(-1, keepdims=True)
    _psf /= _psf.max()
    kernel1 = np.ones((5, 5), np.float32) / 25
    _psf = cv2.filter2D(src=_psf, ddepth=-1, kernel=kernel1)

    max_x, max_y = np.max(_psf, axis=0), np.max(_psf, axis=1)

    # smooth the max functions
    smooth_by = int(len(max_x) * .025)
    smooth_max_x = np.convolve(max_x, np.array([1] * smooth_by) / smooth_by, mode='same')
    smooth_by = int(len(max_y) * .025)
    smooth_max_y = np.convolve(max_y, np.array([1] * smooth_by) / smooth_by, mode='same')

    # correct boundaries
    a = int(np.ceil(smooth_by))
    b = int(np.floor(smooth_by))
    smooth_max_x[:a] = smooth_max_x[a]
    smooth_max_x[-b:] = smooth_max_x[-b]
    smooth_max_y[:a] = smooth_max_y[a]
    smooth_max_y[-b:] = smooth_max_y[-b]

    # get derivative
    dmax_x = smooth_max_x[smooth_by:] - smooth_max_x[:-smooth_by]
    dmax_y = smooth_max_y[smooth_by:] - smooth_max_y[:-smooth_by]

    dmax_x_mean, dmax_x_std = np.mean(dmax_x), np.std(dmax_x)
    dmax_y_mean, dmax_y_std = np.mean(dmax_y), np.std(dmax_y)

    # count peaks in derivative
    # x_out = np.logical_or(dmax_x > dmax_x_mean + dmax_x_std, dmax_x < dmax_x_mean - dmax_x_std).astype(int)
    # y_out = np.logical_or(dmax_y > dmax_y_mean + dmax_y_std, dmax_y < dmax_y_mean - dmax_y_std).astype(int)

    argmax_x, argmax_y = np.argmax(dmax_x) + smooth_by / 2, np.argmax(dmax_y) + smooth_by / 2
    argmin_x, argmin_y = np.argmin(dmax_x) + smooth_by / 2, np.argmin(dmax_y) + smooth_by / 2
    if np.min(dmax_x) > dmax_x_mean - dmax_x_std:  # corner cases
        argmin_x = argmax_x
    if np.min(dmax_y) > dmax_y_mean - dmax_y_std:  # corner cases
        argmin_y = argmax_y
    if np.max(dmax_x) < dmax_x_mean + dmax_x_std:  # corner cases
        argmax_x = argmin_x
    if np.max(dmax_y) < dmax_y_mean + dmax_y_std:  # corner cases
        argmax_y = argmin_y
    meanx = (argmax_x + argmin_x) / 2
    meany = (argmax_y + argmin_y) / 2

    if plot:
        fig = plt.figure(figsize=(10, 5), num=1, clear=True)
        plt.subplot(121)
        plt.plot(max_x - np.mean(max_x), alpha=0.1), plt.plot(smooth_max_x - np.mean(smooth_max_x)), plt.plot(dmax_x)
        plt.axhline(dmax_x_mean - dmax_x_std, linestyle='--'), plt.axhline(dmax_x_mean + dmax_x_std, linestyle='--')
        plt.axvline(argmin_x, alpha=0.1), plt.axvline(argmax_x, alpha=0.1), plt.axvline(meanx, alpha=0.5),
        plt.subplot(122)
        plt.plot(max_y - np.mean(max_y), alpha=0.1), plt.plot(smooth_max_y - np.mean(smooth_max_y)), plt.plot(dmax_y)
        plt.axhline(dmax_y_mean - dmax_y_std, linestyle='--'), plt.axhline(dmax_y_mean + dmax_y_std, linestyle='--')
        plt.axvline(argmin_y, alpha=0.1), plt.axvline(argmax_y, alpha=0.1), plt.axvline(meany, alpha=0.5),
        plt.title(f"{psf_f[-20:]}"), plt.savefig(os.path.join(save_loc, 'plots', os.path.basename(psf_f)))

    var_x = np.mean(((np.arange(len(max_x)) - meanx) ** 2) * max_x / np.sum(max_x))
    var_y = np.mean(((np.arange(len(max_y)) - meany) ** 2) * max_y / np.sum(max_y))
    std_x, std_y = max(5, np.sqrt(var_x)), max(5, np.sqrt(var_y))
    # print(f"stdx {std_x} stdy {std_y} meanx {meanx} meany {meany}")
    # ret = [[max(int(meanx - n * std_x), 0), max(int(meany - n * std_y), 0)],
    #        [min(int(meanx + n * std_x), _psf.shape[1]), min(int(meany + n * std_y), _psf.shape[0])]]
    ret = [[int(meanx - n * std_x), int(meany - n * std_y)],
           [int(meanx + n * std_x), int(meany + n * std_y)]]
    if return_data:
        ret = ret, dmax_x, dmax_y
    return ret

def showcrop(crop, params):
    global boxes
    if skip_windows:
        k = ord('r')
    else:
        while True:
            cv2.imshow('crop', crop)
            k = cv2.waitKey(20)
            if k == ord('r'):
                break
    if ord('r') == k:
        p = os.path.join(params[2], 'crop', f'{params[3]},{params[4]}.tiff')
        # median filtering if needed
        if median_filter_size > 0:
            crop = scipy.signal.medfilt(crop, [median_filter_size, median_filter_size, 1])

        cv2.imwrite(p, cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2BGR))  # cv2 need BGR
        # print(f"Written to file {p}")
        if not skip_windows:
            cv2.destroyWindow('crop')

        psfs[len(psfs)] = np.copy(crop).astype(np.float32) / 255
        # x1 x2 -y1 -y2
        h, w = metadata['sensor_res']
        dh, dw = metadata['pixel_size']
        # range [left, right, bottom, top]
        # center [x = (top+bottom)/2, y=(left+right)/2]
        # boxes [[left, top], [right, bottom]]

        metadata['center'].append([(boxes[0][1] + boxes[1][1])/2 * fac, (boxes[0][0] + boxes[1][0])/2 * fac])
        metadata['maxr'] = max((np.array(metadata['position_index']) ** 2).sum(1) ** 0.5)
        metadata['max_angle'] = abs(np.arctan(metadata['maxr'] / metadata['z']))
        metadata['ndata'] += 1

def on_mouse(event, x, y, flags, params):
    global drawing, ix, iy, cx, cy, to_view, boxes, psf, fac
    global metadata, psfs
    t = time()
    if event == cv2.EVENT_MOUSEMOVE:
        cx, cy = x, y
        if drawing:
            img = np.copy(params[1])
            # # For Drawing Line
            # cv2.line(img, pt1=(ix, iy), pt2=(x, y), color=(255, 255, 255), thickness=3)
            # ix = x
            # iy = y
            # For Drawing Rectangle
            cv2.rectangle(img, pt1=(ix, iy), pt2=(x, y), color=(255, 255, 255), thickness=3)
            to_view = img
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Start Mouse Position: ' + str(x) + ', ' + str(y))
        sbox = [x, y]
        boxes = [sbox]
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        print('End Mouse Position: ' + str(x) + ', ' + str(y))
        boxes.append([x, y])
        drawing = False
        crop = psf[boxes[-2][1] * fac:boxes[-1][1] * fac, boxes[-2][0] * fac:boxes[-1][0] * fac]

        showcrop(crop, params)

def load_psf(filename, backlight):
    _psf = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if _psf is None:
        _psf = np.load(filename)
    if force_grayscale:
        if _psf.shape[-1] == 3:
            _psf = cv2.cvtColor(_psf, cv2.COLOR_RGB2GRAY)
        if _psf.shape[-1] == 4:
            _psf = cv2.cvtColor(_psf, cv2.COLOR_RGBA2GRAY)
    else:
        _psf = cv2.cvtColor(_psf, cv2.COLOR_BGR2RGB)
    psf_curr = _psf.astype(np.float32) - backlight.astype(np.float32)
    psf_curr[_psf < psf_curr] = 0  # correct overflows
    psf_curr[psf_curr < 0] = 0
    return psf_curr

if __name__ == "__main__":
    """
    R: 620-750 nm  685
    G: 495-570 nm  532.8
    B: 450-495 nm  472.5
    """
    loc = "/depot/chan129/data/Deconvolution/PSFs/quadratic_color_psfs_5db_updated"
    grid_size = [20, 20]  # initialize with 0,0 if you want to extract from file name
    do_normalize_exposure = False
    source_dist = 1  # 4.52  # 25.4 * 0.8
    setup_data = {
        "pixel_size": (0.002, 0.002), 
        "z": -1000, 
        "wv": 520
    }
    skip_windows = True
    force_grayscale = False
    median_filter_size = -1  # <= 0 if not needed
    n = 15

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--n_channels', type=int, default=3)
    parser.add_argument('--max_psf_size', type=int, default=128)
    parser.add_argument('--save_dir', default="")
    args = parser.parse_args()
    name = os.path.basename(loc)

    if args.save_dir == "":
        save_loc = f"/scratch/gilbreth/wweligam/dataset/{name}"
    else:
        save_loc = args.save_dir

    # initialize. load backlight images
    psf_filenames = sorted(glob.glob(join(loc, "*.tiff")))
    print(f"Found {len(psf_filenames)} PSF files")
    if len(psf_filenames) == 0:
        psf_filenames = sorted(glob.glob(join(loc, "*.npy")))
        psf = np.load(psf_filenames[0])
    else:
        psf = cv2.imread(psf_filenames[0], cv2.IMREAD_UNCHANGED)

    try:
        backlight1 = cv2.imread(join(loc, "backlight", "backlight1.tiff"), cv2.IMREAD_UNCHANGED)[:, :, :3]
        backlight2 = cv2.imread(join(loc, "backlight", "backlight2.tiff"), cv2.IMREAD_UNCHANGED)[:, :, :3]
        backlight = (backlight1.astype(np.float32) + backlight2.astype(np.float32))/2
    except:
        backlight = cv2.imread(join(loc, "backlight", "backlight.tiff"), cv2.IMREAD_UNCHANGED)[:, :, :3]
    backlight = cv2.cvtColor(backlight, cv2.COLOR_BGR2RGB)

    # extracting data from filenames
    exposure_min = 1e10
    exposure = -1
    for psf_f in psf_filenames:
        data = extract_filename(psf_f)
        if 'exposure' in data.keys():
            exposure = data['exposure']
            exposure_min = min(exposure_min, exposure)
        grid_size[0] = max(grid_size[0], data['x'])
        grid_size[1] = max(grid_size[1], data['y'])
    if exposure_min == 1e10:
        do_normalize_exposure = False
    if do_normalize_exposure:
        print(f"Normalize exposure by {exposure_min:.4f}")
        metadata['exposure_norm'] = exposure_min

    # create folders
    generate_folder(save_loc)
    if os.path.exists(join(save_loc, 'crop')):
        shutil.rmtree(join(save_loc, 'crop'))
    generate_folder(join(save_loc, 'crop'))
    generate_folder(join(save_loc, 'results'))
    generate_folder(join(save_loc, 'plots'))

    # setup psf metadata
    for key in setup_data:
        metadata[key] = setup_data[key]
    metadata['n_rays'] = 'inf'
    metadata['ndata'] = 0
    metadata['sensor_res'] = psf.shape[:2]
    metadata['sensor_size'] = np.array(setup_data['pixel_size']) * np.array(metadata['sensor_res'])
    metadata['is_psf_in_sensor_size'] = 0
    metadata['coordinates'] = 'cartesian'
    metadata['grid_size'] = grid_size
    print(metadata)
    print(setup_data)
    
    psfsim = PSFSimulator(image_shape=metadata['sensor_res'], psf_size=args.max_psf_size, normalize=True)
    # generate PSFs
    # generated_psf, psf_metadata = psfsim.generate_all_psfs(metadata['grid_size'][0], metadata['grid_size'][1], outc=1, inc=args.n_channels, **kwargs)
    psfs_all_sum = None
    for i_psf, psf_f in enumerate(psf_filenames):
        """
        q: quit
        w: increase size
        s: decrease size

        r: write cropped to file
        """
        print(f"Processing {psf_f}")
        psf = load_psf(psf_f, backlight)
        if psfs_all_sum is None:
            psfs_all_sum = psf.astype(np.float32)
        else:
            psfs_all_sum += psf.astype(np.float32)

        resized = cv2.resize(psf, (psf.shape[1] // fac, psf.shape[0] // fac))

        # get x y position from filename
        data = extract_filename(psf_f)
        _x = data['x']
        _y = data['y']
        if 'exposure' in data.keys():
            exposure = data['exposure']
        if _x == -1 or _y == -1:
            _x = i_psf // grid_size[0]
            _y = i_psf % grid_size[0]
        metadata['position_index'].append([_x,_y])
        metadata['index'].append(i_psf)

        # estimate initial box from image data
        boxes = estimate_box(resized, n=n)  # global
        xs, xe, ys, ye = boxes[-2][1] * fac, boxes[-1][1] * fac, boxes[-2][0] * fac, boxes[-1][0] * fac
        ps, pe, qs, qe = max(0, xs), min(xe, psf.shape[0]), max(0, ys), min(ye, psf.shape[1])
        crop = np.zeros((xe-xs, ye-ys, 3), dtype=psf.dtype)
        crop[max(0, -xs): xe-xs + min(0, psf.shape[0] - xe), max(0, -ys): ye-ys+min(0, psf.shape[1] - ye)] = psf[ps:pe, qs:qe]
        # print(f"crop shape {xs,xe,ys,ye} {crop.shape}")
        if do_normalize_exposure:
            crop = (crop.astype(float) * exposure_min / exposure).astype(crop.dtype)
        
        # plot boxes
        fig, ax = plt.subplots(num=2, clear=True)
        ax.imshow(resized/resized.max())
        rect = patches.Rectangle((boxes[0][0], boxes[0][1]), abs(boxes[0][0] - boxes[1][0]), abs(boxes[0][1] - boxes[1][1]),
                                 linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        plt.savefig(join(save_loc, 'plots', "_" + os.path.basename(psf_f)))

        # show and store cropped psf
        params = [psf_f, resized, save_loc, _x, _y]
        showcrop(crop, params)

        # for mouse cropping
        img = np.copy(resized)
        cv2.rectangle(img, pt1=boxes[0], pt2=boxes[1], color=(255, 255, 255), thickness=3)
        to_view = img
        while True:
            if skip_windows:
                k = ord('q')
            else:
                cv2.namedWindow(psf_f)
                cv2.setMouseCallback(psf_f, on_mouse, params)
                cv2.imshow(psf_f, to_view)
                k = cv2.waitKey(20)
            if k == ord('q'):
                boxes = []
                break
            if k == ord('w'):
                fac -= 1
                resized = cv2.resize(psf, (psf.shape[1] // fac, psf.shape[0] // fac))
                to_view = np.copy(resized)
                params[1] = resized
                print(fac)
            if k == ord('s'):
                fac += 1
                resized = cv2.resize(psf, (psf.shape[1] // fac, psf.shape[0] // fac))
                to_view = np.copy(resized)
                params[1] = resized
                print(fac)
        cv2.destroyAllWindows()

    # sort the stored psfs as shape (inc, outc, gridh, gridw, h, w)
    generated_psf = [[None]*grid_size[1] for _ in range(grid_size[0])]
    psf_metadata = [[None]*grid_size[1] for _ in range(grid_size[0])]
    max_h, max_w = 0, 0
    for i in range(metadata['ndata']):
        x, y = metadata['position_index'][i]
        generated_psf[x][y] = psfs[i]
        psf_metadata[x][y] = metadata['center'][i]
        max_h = max(max_h, psfs[i].shape[0])
        max_w = max(max_w, psfs[i].shape[1])
    print(f"Max h {max_h} w {max_w}, Max psf size {args.max_psf_size}")
    max_psf_size = min(max(max_h, max_w), args.max_psf_size)
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            generated_psf[x][y] = crop_arr(einops.rearrange(generated_psf[x][y],'h w c -> c h w'), max_psf_size, max_psf_size)
            print(generated_psf[x][y].shape)
    generated_psf = einops.repeat(np.array(generated_psf), f'x y c h w -> c 1 x y h w')
    psf_metadata = einops.repeat(np.array(psf_metadata), 'x y d -> 1 1 d x y')
    metadata['psf_metadata'] = psf_metadata
    
    # setup homographies TODO: Check if the homographies are correct
    homography = np.zeros([*metadata['sensor_res'], 3, 3])
    for x in range(grid_size[0]-1):
        for y in range(grid_size[1]-1):
            pts_src = np.array([[x, y], [x+1, y], [x, y+1], [x+1, y+1]], dtype=np.float32)
            pts_dst = np.array([psf_metadata[0,0,:2,int(_x),int(_y)] for (_x,_y) in pts_src], dtype=np.float32)
            # Calculate the homography matrix
            H, _ = cv2.findHomography(pts_src, pts_dst)
            # Find the pixels inside the polygon
            mask = np.zeros(metadata['sensor_res'], dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts_dst.astype(int), 255)
            x_idx, y_idx = np.where(mask == 255)
            # Set H for all pixels inside the polygon
            for i in range(len(x_idx)):
                homography[x_idx[i], y_idx[i]] = H
    psfsim.set_H_obj(horizontal_distance=source_dist, vertical_distance=source_dist, horizontal_translation=0, vertical_translation=0)
    psfsim.set_H_img(homography=homography)

    # saving npy
    np.save(f"{save_loc}/psfs_all_sum_{name}", psfs_all_sum/len(psf_filenames)/255)
    PSFSimulator.display_psfs(image_shape=metadata['sensor_res'], psfs=generated_psf, weights=None, metadata=psf_metadata, title="All PSFs", skip=1)
    plt.savefig(f"{save_loc}/psfs.png")
    PSFSimulator.display_psfs(image_shape=metadata['sensor_res'], psfs=generated_psf, weights=None, metadata=psf_metadata, title="10x10 PSFs", skip=generated_psf.shape[2] // 10)
    plt.savefig(f"{save_loc}/psfs_10x10.png")
    
    # saving PSFs
    psfsim.save_psfs(save_loc, generated_psf, metadata=metadata, H_img=psfsim.H_img, H_obj=psfsim.H_obj)

    # check loading
    print("Loading PSFs")
    psfs, metadata = psfsim.load_psfs(save_loc)
    for name in psfs:
        print(f"{name} {psfs[name].shape}")
