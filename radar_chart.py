"""
Radar chart comparing selected deblurring methods across PSNR, SSIM, LPIPS, FID, and NIQE.
Metrics are normalized to [0, 1] so that "better" is always 1 (higher = better for PSNR/SSIM;
lower = better for LPIPS/FID/NIQE, so we use 1 - normalized for those).
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Table data: Method name -> (PSNR, SSIM, LPIPS, FID, NIQE)
# Extracted from process_table.py; strip LaTeX commands for numeric values
def parse_table(table_code):
    """Parse LaTeX table and return list of (method_name, PSNR, SSIM, LPIPS, FID, NIQE)."""
    def clean_cell(text):
        text = re.sub(r"\\(RR|BB|RB|textbf)\{([^}]*)\}", r"\2", text)
        return re.sub(r"[^0-9.\-]", "", text.strip())

    lines = table_code.strip().split("\n")
    header_idx = -1
    for i, line in enumerate(lines):
        if r"\textbf{Method}" in line:
            header_idx = i
            break
    if header_idx == -1:
        return []

    # Column order in table: Method, PSNR, SSIM, LPIPS, DISTS, FID, NIQE, ...
    # We want indices: 0=Method, 1=PSNR, 2=SSIM, 3=LPIPS, 5=FID, 6=NIQE
    want_cols = (1, 2, 3, 4, 5, 6, 7, 8, 9)  # PSNR, SSIM, LPIPS, DISTS, FID, NIQE, MUSIQ, MANIQA, CLIPIQA
    rows = []
    for i in range(header_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith(r"\bottomrule") or line.startswith(r"\toprule"):
            continue
        if r"\hline" in line:
            line = line.replace(r" \hline", "").replace(r"\\ \hline", "")
        if "&" not in line:
            continue
        parts = [p.strip() for p in line.split("&")]
        if len(parts) < 7:
            continue
        method = re.sub(r"\\cite\{[^}]+\}", "", parts[0]).strip()
        method = method.replace(r"\\", "").strip()
        try:
            vals = [float(clean_cell(parts[j])) for j in want_cols]
            rows.append((method, *vals))
        except ValueError:
            continue
    return rows


# Same table as in process_table.py (subset of columns used)
TABLE_CODE = r"""
    \toprule
    \textbf{Method} & \textbf{PSNR$\uparrow$} & \textbf{SSIM$\uparrow$} & \textbf{LPIPS$\downarrow$} & \textbf{DISTS$\downarrow$} & \textbf{FID$\downarrow$} & \textbf{NIQE$\downarrow$} & \textbf{MUSIQ$\uparrow$} & \textbf{MANIQA$\uparrow$} & \textbf{CLIPIQA$\uparrow$}  \\ \hline
    Reference           & 29.87       & 0.8841        & 0.3018        & 0.2512        & 39.41       & 9.6690        & 23.40       & 0.3160        & 0.4502 \\ \hline
    FBANet\cite{wei2023fbanet}     & 30.88       & 0.9341        & 0.1891        & 0.1354        & 21.74       & 6.5970        & 34.53       & 0.3153        & \RB{0.4864} \\
    Burstormer\cite{dudhane2023burstormer}  & 35.07       & 0.9454        & 0.0948        & 0.0669        & 12.32       & \BB{5.5131}   & 36.72       & 0.3450        & 0.3600 \\
    MPRNet\cite{zamir2021mprnet}      & 35.18       & 0.9471        & 0.1309        & 0.1045        & 21.51       & 6.9476        & 38.93       & 0.3540        & \BB{0.4471} \\
    Restormer \cite{zamir2022restormer}    & 35.24       & \RB{0.9522}   & 0.1172        & 0.0967        & 16.10       & 7.0328        & \BB{39.09}  & \BB{0.3570}   & 0.4409 \\
    GAN (NAFNet) \cite{chen2022nafnet} & \RB{38.13}  & \BB{0.9500}   & \RB{0.0632}   & \RB{0.0319}   & \RB{6.97}   & \RB{4.7797}   & \RB{39.25}  & \RB{0.3750}   & 0.4396 \\
    GAN (SwinU) \cite{liu2021swin} & \BB{36.04}  & 0.9471        & \BB{0.0837}   & \BB{0.0549}   & \BB{11.83}  & 5.6657        & 38.04       & 0.3468        & 0.3896 \\ \hline
    
    ResShift (NAFNet)  \cite{yue2023resshift} & 26.12       & 0.8623        & 0.2927        & 0.1760        & 41.82       & 6.1530        & 29.32       & 0.3079        & 0.3429 \\
    InDI (NAFNet) \cite{delbracio2023inversion} & \RB{32.69}  & 0.8987        & 0.2486        & 0.2134        & 57.12       & 5.4929        & 28.41       & 0.3162        & 0.3207 \\
    VSD \cite{wu2024neurips_osediff} & 29.75       & 0.9019        & 0.1232        & 0.0733        & 20.02       & 5.2592        & \BB{39.08}  & \BB{0.3621}   & 0.3935 \\
    Ours         & \BB{31.43}  & \RB{0.9052}   & \RB{0.0984}   & \RB{0.0585}   & \RB{14.22}  & \RB{5.1981}   & \RB{39.20}  & \RB{0.3631}   & \RB{0.4184} \\
    
\bottomrule
"""

# Which methods to show on the radar (edit this list to select methods)
SELECTED_METHODS = [
    # "Reference",
    "FBANet",
    # "Burstormer",
    "MPRNet",
    "Restormer",
    # "GAN (NAFNet)",
    # "GAN (SwinU)",
    # "Ours (w/o HF loss)",
    "ResShift (NAFNet)",
    "InDI (NAFNet)",
    "VSD",
    "Ours",
]

METRIC_NAMES = ["PSNR", "SSIM", "LPIPS", "DISTS", "FID", "NIQE", "MUSIQ", "MANIQA", "CLIPIQA"]
MAX_METRIC_VALUES = [40.0, 1.0, 0.35, 0.3, 60.0, 10.0, 40.0, 0.4, 0.5]
# For each metric: True = higher is better, False = lower is better
HIGHER_IS_BETTER = [True, True, False, False, False, False, True, True, True]


def normalize_for_radar(data_rows):
    """data_rows: list of (method, PSNR, SSIM, LPIPS, FID, NIQE). Returns (methods, matrix)."""
    methods = [r[0] for r in data_rows]
    arr = np.array([[r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]] for r in data_rows])
    n_metrics = arr.shape[1]
    normalized = np.zeros_like(arr)
    for j in range(n_metrics):
        col = arr[:, j]
        lo, hi = col.min(), col.max()
        if lo == hi:
            normalized[:, j] = 1.0
        elif HIGHER_IS_BETTER[j]:
            lo=0.0
            normalized[:, j] = (col - lo) / (hi - lo)
        else:
            hi = MAX_METRIC_VALUES[j]
            # lower is better -> map to 1 at min, 0 at max
            normalized[:, j] = (hi - col) / (hi - lo)
    return methods, normalized


def plot_radar(methods, values, metric_names, out_path="radar_chart.pdf"):
    """Draw radar chart and save to out_path."""
    n_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for i, (method, row) in enumerate(zip(methods, values)):
        vals = row.tolist()
        vals += vals[:1]
        if method == "Ours":
            ax.plot(angles, vals, linewidth=2, label=method, color="red", linestyle="--")
            # no fill for Ours
        else:
            ax.plot(angles, vals, linewidth=2, label=method, color=colors[i])
            ax.fill(angles, vals, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])  # no radial circle labels
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=12, ncol=2, frameon=True)
    # ax.set_title("Method comparison (normalized: 1 = best)", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Radar chart: PSNR, SSIM, LPIPS, FID, NIQE")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Method names to include (default: use SELECTED_METHODS)")
    parser.add_argument("--out", default="radar_chart.pdf", help="Output path (default: radar_chart.pdf)")
    args = parser.parse_args()

    rows = parse_table(TABLE_CODE)
    if not rows:
        print("No rows parsed from table.")
        return

    selected = args.methods if args.methods else SELECTED_METHODS
    # Normalize names for matching (strip trailing space and cite)
    def norm(s):
        return re.sub(r"\\cite\{[^}]+\}", "", s).strip()
    data_rows = []
    seen = set()
    for r in rows:
        name = norm(r[0])
        print(name, r)
        if name in seen:
            continue
        for sel in selected:
            if name == sel or name.startswith(sel + " ") or name.startswith(sel + "\\"):
                data_rows.append((name, *r[1:]))
                seen.add(name)
                break

    if not data_rows:
        print("No selected methods found. Available:", [r[0] for r in rows])
        return

    methods, normalized = normalize_for_radar(data_rows)
    plot_radar(methods, normalized, METRIC_NAMES, out_path=args.out)


if __name__ == "__main__":
    main()
