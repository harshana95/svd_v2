import re

def highlight_best_and_second(latex_table, best_cmd=r"\RB", second_best_cmd=r"\BB", third_best_cmd=r"\TB", by_section=True):
    """
    by_section: If True (default), treat each block between \\hline/\\bottomrule separately.
                If False, treat the whole table as one block for best/second-best per column.
    """
    # Removes existing highlighting to avoid nested tags or parsing errors
    def clean_cell_tags(text):
        pattern = r"\\(RR|BB|RB|TB|textbf)\{([^}]*)\}"
        while re.search(pattern, text):
            text = re.sub(pattern, r"\2", text)
        return text

    lines = latex_table.strip().split('\n')
    
    # 1. Identify header row and determine metric directions
    header_idx = -1
    directions = []
    for i, line in enumerate(lines):
        if r'\textbf{Method}' in line:
            header_idx = i
            row_content = line.split('\\\\')[0].strip()
            cols = [c.strip() for c in row_content.split('&')]
            for col in cols[1:]: # Skip the 'Method' column
                if r'\uparrow' in col: directions.append('max')
                elif r'\downarrow' in col: directions.append('min')
                else: directions.append(None)
            break
            
    if header_idx == -1: return "Error: Could not find header row."

    # 2. Identify subsections (or one block if by_section=False)
    sections = []
    current_block = []
    for i in range(header_idx + 1, len(lines)):
        line = lines[i].strip()
        if by_section and any(cmd in line for cmd in [r'\hline', r'\bottomrule']):
            if '&' in line:
                current_block.append(i)  # include row that ends with \hline
            if current_block:
                sections.append(current_block)
                current_block = []
        elif '&' in line:
            current_block.append(i)

    if not by_section and current_block:
        sections = [current_block]  # whole table as one block

    # 3. Process each section (or the single block)
    for section_indices in sections:
        # First pass: remove existing commands from all cells so we work with clean content
        for idx in section_indices:
            line = lines[idx]
            split_line = line.split('\\\\', 1)
            row_content = split_line[0].strip()
            suffix = (' \\\\' + split_line[1]) if len(split_line) > 1 else ''
            parts = [p.strip() for p in row_content.split('&')]
            # Keep first column (Method) as-is; clean data cells
            cleaned_parts = [parts[0]] + [clean_cell_tags(p) for p in parts[1:]]
            lines[idx] = '&'.join(cleaned_parts) + suffix

        grid = []
        for idx in section_indices:
            row_content = lines[idx].split('\\\\')[0].strip()
            cols = [c.strip() for c in row_content.split('&')]
            row_vals = []
            for col_val in cols[1:]:
                try:
                    clean_val = re.sub(r'[^0-9\.\-]', '', col_val)
                    row_vals.append(float(clean_val))
                except ValueError:
                    row_vals.append(None)
            grid.append(row_vals)
            
        if not grid: continue

        num_cols = len(grid[0])
        for j in range(num_cols):
            dir_type = directions[j]
            if dir_type is None: continue
            
            # Get values and their original row indices within this block
            col_data = []
            for row_idx_in_block, row in enumerate(grid):
                if row[j] is not None:
                    col_data.append((row[j], row_idx_in_block))
            
            if not col_data: continue
            
            # Sort: Descending for max (arrows up), Ascending for min (arrows down)
            col_data.sort(key=lambda x: x[0], reverse=(dir_type == 'max'))
            
            best_val = col_data[0][0]
            # Find second and third best (next distinct values)
            second_best_val = col_data[1][0] if len(col_data) > 1 else None
            third_best_val = col_data[2][0] if len(col_data) > 2 else None
            second_best_assigned = False
            third_best_assigned = False

            # Apply highlighting to the original lines
            for val, row_idx_in_block in col_data:
                line_idx = section_indices[row_idx_in_block]
                parts = lines[line_idx].split('&')
                cell = parts[j + 1].strip()
                # Handle LaTeX line endings (e.g. " 0.4864 \\ \hline")
                if '\\\\' in cell:
                    idx_ee = cell.find('\\\\')
                    clean_content = clean_cell_tags(cell[:idx_ee].strip())
                    suffix = cell[idx_ee:]
                else:
                    clean_content = clean_cell_tags(cell.strip())
                    suffix = ''
                
                if val == best_val:
                    parts[j + 1] = f"{best_cmd}{{{clean_content}}}{suffix}"
                elif second_best_val is not None and val == second_best_val and not second_best_assigned:
                    parts[j + 1] = f"{second_best_cmd}{{{clean_content}}}{suffix}"
                    second_best_assigned = True
                elif third_best_val is not None and val == third_best_val and not third_best_assigned:
                    parts[j + 1] = f"{third_best_cmd}{{{clean_content}}}{suffix}"
                    third_best_assigned = True
                else:
                    parts[j + 1] = f"{clean_content}{suffix}"
                lines[line_idx] = "&".join(parts)
                    
    return '\n'.join(lines)
# --- Example Usage ---
table_code = r"""
\toprule
    \textbf{Method} & \textbf{PSNR$\uparrow$} & \textbf{SSIM$\uparrow$} & \textbf{LPIPS$\downarrow$} & \textbf{DISTS$\downarrow$} & \textbf{FID$\downarrow$} & \textbf{NIQE$\downarrow$} & \textbf{MUSIQ$\uparrow$} & \textbf{MANIQA$\uparrow$} & \textbf{CLIPIQA$\uparrow$}  \\ \hline
    Yang \emph{et al.}~\cite{parfocal} & 15.0096 & 0.4429 & 0.5428 & 0.3385 & 192.3649 & 5.9950 & 37.0225 & 0.2027 & 0.2403 \\
    Tseng \emph{et al.}~\cite{Tseng2021NeuralNanoOptics} & 24.5606 & 0.8032 & 0.3010 & 0.1937 & 124.9688 & 4.7498 & 46.9808 & 0.2378 & 0.3925 \\
    Pinilla \emph{et al.}~\cite{Pinilla2023} & 24.5414 & 0.7069 & 0.3867 & 0.2112 & 163.5929 & 5.2301 & 35.2180 & 0.2091 & 0.2659 \\
    Ours  & 21.9542 & 0.6294 & 0.2042 & 0.1397 & 108.9117 & 3.9841 & 61.0900 & 0.3747 & 0.5121\\
\bottomrule
"""
# GAN (NAFNet) \cite{chen2022nafnet} &  \RB{38.13}   & \BB{0.9500}  &   \RB{0.0632} &  \RB{0.0319}      & \RB{6.97}     &  \RB{4.7797}    & \RB{39.25}   &   \RB{0.3750}  &  0.4396 \\
#     GAN (SwinU) \cite{liu2021swin}    & \BB{36.04}  & 0.9471  & \BB{0.0837}   & \BB{0.0549}   & \BB{11.83}  & 5.6657  & 38.04 & 0.3468  & 0.3896 \\

# By default, each block between \hline is processed separately (by_section=True)
print(highlight_best_and_second(table_code))
# To treat the whole table as one block for best/second per column:
print(highlight_best_and_second(table_code, by_section=False))