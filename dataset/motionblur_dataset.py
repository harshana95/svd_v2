import os
import glob
import random

from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

class BaseClass(Dataset):
    def __init__(self, opt):
        self.height, self.width = opt.image_size
        self.data_dir = opt.data_dir
        self.image_size = opt.image_size
        self.length = 0
        
    def __len__(self):
        return self.length
        

    def load_frames(self, frames):
        # frames: numpy array of shape (N, H, W, C), 0–255
        # → tensor of shape (N, C, H, W) as float
        pixel_values = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float()
        # normalize to [-1, 1]
        pixel_values = pixel_values / 127.5 - 1.0
        # resize to (self.height, self.width)
        pixel_values = F.interpolate(
            pixel_values,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False
        )
        return pixel_values
        
    def get_batch(self, idx):
        raise Exception('Get batch method is not realized.')

    def __getitem__(self, idx):
        while True:
            try:
                video, caption, motion_blur = self.get_batch(idx)
                break
            except Exception as e:
                print(e)
                idx = random.randint(0, self.length - 1)
            
        video, = [
            resize_for_crop(x, self.height, self.width) for x in [video]
        ] 
        video, = [
            transforms.functional.center_crop(x, (self.height, self.width)) for x in [video]
        ]
        data = {
            'video': video, 
            'caption': caption, 
        }
        return data

class FullMotionBlurDataset(BaseClass):
    """
    A dataset that randomly selects among 1x, 2x, or large-blur modes per sample.
    """
    def __init__(self, opt):
        super().__init__(opt)
        self.seq_dirs = []
        
        # TRAIN/VAL: build seq_dirs from each category's list or fallback
        for category in sorted(os.listdir(self.data_dir)):  # each dataset
            cat_dir = os.path.join(self.data_dir, category)
            if not os.path.isdir(cat_dir):
                continue
            
            fps_root = os.path.join(cat_dir, 'lower_fps_frames')
            if os.path.isdir(fps_root):
                for seq in sorted(os.listdir(fps_root)):
                    seq_path = os.path.join(fps_root, seq)
                    if os.path.isdir(seq_path):
                        self.seq_dirs.append(seq_path)

        L = len(self.seq_dirs)
        self.seq_dirs = self.seq_dirs[int(opt.split[0]*L):int(opt.split[1]*L)]
        assert self.seq_dirs, f"No sequences found  in {self.data_dir}"
        print(f"Initialized dataset with {len(self.seq_dirs)} sequences")

    def __len__(self):
        return len(self.seq_dirs)

    def __getitem__(self, idx):
        # Prepare base items
        seq_dir = self.seq_dirs[idx]
        frame_paths = sorted(glob.glob(os.path.join(seq_dir, '*.png')))
        mode = random.choice(['1x', '2x', 'large_blur'])
        # breakpoint()
        if mode == '1x' or len(frame_paths) < 50:
            base_rate = random.choice([1, 2])
            blur_img, seq_frames, inp_int, out_int, _ = generate_1x_sequence(
                frame_paths, window_max=16, output_len=17, base_rate=base_rate
            )
        elif mode == '2x':
            base_rate = random.choice([1, 2])
            blur_img, seq_frames, inp_int, out_int, _ = generate_2x_sequence(
                frame_paths, window_max=16, output_len=17, base_rate=base_rate
            )
        else:
            max_base = min((len(frame_paths) - 1) // 17, 3)
            base_rate = random.randint(1, max_base)
            blur_img, seq_frames, inp_int, out_int, _ = generate_large_blur_sequence(
                frame_paths, window_max=16, output_len=17, base_rate=base_rate
            )
        file_name = frame_paths[0]
        num_frames = 16

        # blur_img is a single frame; wrap in batch dim
        blur_input = self.load_frames(np.expand_dims(blur_img, 0))
        # seq_frames is list of frames; stack along time dim
        video = self.load_frames(np.stack(seq_frames, axis=0))

        
        relative_file_name = os.path.relpath(file_name, self.data_dir)
        
        data = {
            'file_name': relative_file_name,
            'blur_img':  blur_input,
            'num_frames': num_frames,
            'video':     video,
            'caption':   "",
            'mode':      mode,
            'input_interval':  inp_int,
            'output_interval': out_int,
        }
        return data
    

# helper functions
def build_blur(frame_paths, gamma=2.2):
    """
    Simulate motion blur using inverse-gamma (linear-light) summation:
    - Load each image, convert to float32 sRGB [0,255]
    - Linearize via inverse gamma: linear = (img/255)^gamma
    - Sum linear values, average, then re-encode via gamma: (linear_avg)^(1/gamma)*255
    Returns a uint8 numpy array.
    """
    acc_lin = None
    for p in frame_paths:
        img = np.array(Image.open(p).convert('RGB'), dtype=np.float32)
        # normalize to [0,1] then linearize
        lin = np.power(img / 255.0, gamma)
        acc_lin = lin if acc_lin is None else acc_lin + lin
    # average in linear domain
    avg_lin = acc_lin / len(frame_paths)
    # gamma-encode back to sRGB domain
    srgb = np.power(avg_lin, 1.0 / gamma) * 255.0
    return np.clip(srgb, 0, 255).astype(np.uint8)

def generate_1x_sequence(frame_paths, window_max =16, output_len=17, base_rate=1, start = None):
    """
    1x mode at arbitrary base_rate (units of 1/240s):
      - Treat each output step as the sum of `base_rate` consecutive raw frames.
      - Pick window size W ∈ [1, output_len]
      - Randomly choose start index so W*base_rate frames fit
      - Group raw frames into W groups of length base_rate
      - Build blur image over all W*base_rate frames for input
      - For each group, build a blurred output frame by summing its base_rate frames
      - Pad sequence of W blurred frames to output_len by repeating last blurred frame
      - Input interval always [-0.5, 0.5]
      - Output intervals reflect each group’s coverage within [-0.5,0.5]
    """
    N = len(frame_paths)
    max_w = min(output_len, N // base_rate)
    max_w = min(max_w, window_max)
    W = random.randint(1, max_w)
    if start is not None:
        # choose start so that W*base_rate frames fit
        assert N >= W * base_rate, f"Not enough frames for base_rate={base_rate}, need {W * base_rate}, got {N}"
    else:
        start = random.randint(0, N - W * base_rate)
        

    # group start indices
    group_starts = [start + i * base_rate for i in range(W)]
    # flatten raw frame paths for blur input
    blur_paths = []
    for gs in group_starts:
        blur_paths.extend(frame_paths[gs:gs + base_rate])
    blur_img = build_blur(blur_paths)

    # build blurred output frames per group
    seq = []
    for gs in group_starts:
        group = frame_paths[gs:gs + base_rate]
        seq.append(build_blur(group))
    # pad with last blurred frame
    seq += [seq[-1]] * (output_len - len(seq))

    input_interval = torch.tensor([[-0.5, 0.5]], dtype=torch.float)
    # each group covers interval of length 1/W
    step = 1.0 / W
    intervals = [[-0.5 + i * step, -0.5 + (i + 1) * step] for i in range(W)]
    num_frames = len(intervals)
    intervals += [intervals[-1]] * (output_len - W)
    output_intervals = torch.tensor(intervals, dtype=torch.float)

    return blur_img, seq, input_interval, output_intervals, num_frames

def generate_2x_sequence(frame_paths, window_max =16, output_len=17, base_rate=1):
    """
    2x mode:
      - Logical window of W output-steps so that 2*W ≤ output_len
      - Raw window spans W*base_rate frames
      - Build blur only over that raw window (flattened) for input
      - before_count = W//2, after_count = W - before_count
      - Define groups for before, during, and after each of length base_rate
      - Build blurred frames for each group
      - Pad sequence of 2*W blurred frames to output_len by repeating last
      - Input interval always [-0.5,0.5]
      - Output intervals relative to window: each group’s center
    """
    N = len(frame_paths)
    max_w = min(output_len // 2, N // base_rate)
    max_w = min(max_w, window_max)
    W = random.randint(1, max_w)
    before_count = W // 2
    after_count = W - before_count
    # choose start so that before and after stay within bounds
    min_start = before_count * base_rate
    max_start = N - (W + after_count) * base_rate
    # ensure we can pick a valid start, else fail
    assert max_start >= min_start, f"Cannot satisfy before/after window for W={W}, base_rate={base_rate}, N={N}"
    start = random.randint(min_start, max_start)


    # window group starts
    window_starts = [start + i * base_rate for i in range(W)]
    # flatten for blur input
    blur_paths = []
    for gs in window_starts:
        blur_paths.extend(frame_paths[gs:gs + base_rate])


    blur_img = build_blur(blur_paths)

    # define before/after group starts
    before_count = W // 2
    after_count = W - before_count
    before_starts = [max(0, start - (i + 1) * base_rate) for i in range(before_count)][::-1]
    after_starts  = [min(N - base_rate, start + W * base_rate + i * base_rate) for i in range(after_count)]

    # all group starts in sequence
    group_starts = before_starts + window_starts + after_starts
    # build blurred frames per group
    seq = []
    for gs in group_starts:
        group = frame_paths[gs:gs + base_rate]
        seq.append(build_blur(group))
    # pad blurred frames to output_len
    seq += [seq[-1]] * (output_len - len(seq))

    input_interval = torch.tensor([[-0.5, 0.5]], dtype=torch.float)
    # each group covers 1/(2W) around its center within [-0.5,0.5]
    half = 0.5 / W
    centers = [((gs - start) / (W * base_rate)) - 0.5 + half
               for gs in group_starts]
    intervals = [[c - half, c + half] for c in centers]
    num_frames = len(intervals)
    intervals += [intervals[-1]] * (output_len - len(intervals))
    output_intervals = torch.tensor(intervals, dtype=torch.float)

    return blur_img, seq, input_interval, output_intervals, num_frames


def generate_large_blur_sequence(frame_paths, window_max=16, output_len=17, base_rate=1):
    """
    Large blur mode (fixed output_len=25) with instantaneous outputs:
      - Raw window spans 25 * base_rate consecutive frames
      - Build blur over that full raw window for input
      - For output sequence:
          • Pick 1 raw frame every `base_rate` (group_starts)
          • Each output frame is the instantaneous frame at that raw index
      - Input interval always [-0.5, 0.5]
      - Output intervals reflect each 1-frame slice’s coverage within the blur window,
        leaving gaps between.
    """
    N = len(frame_paths)
    total_raw = window_max * base_rate
    assert N >= total_raw, f"Not enough frames for base_rate={base_rate}, need {total_raw}, got {N}"
    start = random.randint(0, N - total_raw)

    # build blur input over the full raw block
    raw_block = frame_paths[start:start + total_raw]
    blur_img = build_blur(raw_block)

    # output sequence: instantaneous frames at each group_start
    seq = []
    group_starts = [start + i * base_rate for i in range(window_max)]
    for gs in group_starts:
        img = np.array(Image.open(frame_paths[gs]).convert('RGB'), dtype=np.uint8)
        seq.append(img)
     # pad blurred frames to output_len
    seq += [seq[-1]] * (output_len - len(seq))

    # compute intervals for each instantaneous frame:
    # each covers [gs, gs+1) over total_raw, normalized to [-0.5, 0.5]
    intervals = []
    for gs in group_starts:
        t0 = (gs - start) / total_raw - 0.5
        t1 = (gs + 1 - start) / total_raw - 0.5
        intervals.append([t0, t1])
    num_frames = len(intervals)
    intervals += [intervals[-1]] * (output_len - len(intervals))
    output_intervals = torch.tensor(intervals, dtype=torch.float)

    # input interval
    input_interval = torch.tensor([[-0.5, 0.5]], dtype=torch.float)
    return blur_img, seq, input_interval, output_intervals, num_frames
