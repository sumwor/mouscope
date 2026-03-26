import numpy as np
import caiman as cm
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

def convert_F_to_C_memmap(
    fname_f,
    fname_c=None,
    chunk_frames=200,
    dtype=np.float32
    ):
    """
    Convert a large CaImAn F-order memmap to C-order safely.

    Parameters
    ----------
    fname_f : str
        Input F-order .mmap file
    fname_c : str or None
        Output C-order .mmap file (auto-generated if None)
    chunk_frames : int
        Number of frames processed per chunk
    dtype : numpy dtype
        Data type (must match input)
    """

    Yr, dims, T = cm.load_memmap(fname_f)
    d1, d2 = dims

    if fname_c is None:
        fname_c = fname_f.replace("order_F", "order_C")

    print(f'Converting F → C memmap')
    print(f'Input:  {fname_f}')
    print(f'Output: {fname_c}')
    print(f'Shape:  ({T}, {d1}, {d2})')

    # Create output memmap
    Yc = np.memmap(
        fname_c,
        dtype=dtype,
        mode='w+',
        shape=(T, d1, d2),
        order='C'
    )

    for start in range(0, T, chunk_frames):
        end = min(start + chunk_frames, T)

        # Read F-order chunk
        chunk = (
            Yr[:, start:end]
            .reshape(d1, d2, end - start, order='C')
            .transpose(2, 0, 1)
        )

        Yc[start:end] = chunk
        Yc.flush()

        if start % (chunk_frames * 10) == 0:
            print(f'  written frames {start}–{end}')

    del Yc
    print('Conversion complete.')

    return fname_c

def frame_row_corr_batch(frames, ref0, ref_norm):
    X = frames[:, 1:, :].astype(np.float32)
    X0 = X - X.mean(axis=2, keepdims=True)

    num = np.sum(X0 * ref0[None, :, :], axis=2)
    denom = np.sqrt(np.sum(X0 * X0, axis=2)) * ref_norm[None, :]

    corr = np.where(denom > 0, num / denom, 0)
    return corr.mean(axis=1)

def frame_row_corr(frame, ref):
    """
    Compute average correlation between consecutive rows of a frame.
    Lower correlation → likely shifted rows.
    """
    X = np.array(frame[:], dtype=np.float32, copy=True)
    Y = np.array(ref[:], dtype=np.float32, copy=True)

    X0 = X - X.mean(axis=1, keepdims=True)
    Y0 = Y - Y.mean(axis=1, keepdims=True)

    num = np.sum(X0 * Y0, axis=1)
    denom = np.sqrt(
        np.sum(X0 * X0, axis=1) * np.sum(Y0 * Y0, axis=1)
    )

    corr = np.where(denom > 0, num / denom, 0)
    return float(corr.mean())
    # idx = 9775
    # corr_vals_f1 = frame_row_corr(images[idx,:,:], ref)

    # idx = 1 # normal ref
    # corr_vals_ctrl = frame_row_corr(images[idx,:,:], ref)

    # idx = 27870
    # corr_vals_f2 = frame_row_corr(images[idx,:,:], ref)

    # print(corr_vals_f1)
    # print(corr_vals_f2)
    # print(corr_vals_ctrl)

def rowwise_discontinuity_score(frames, eps=1e-8):
    """
    frames: (T, H, W)
    returns: per-frame min adjacent-row correlation (T,)
    """
    T, H, W = frames.shape
    scores = np.zeros(T, dtype=np.float32)

    for t in range(T):
        F = frames[t].astype(np.float32)

        # normalize rows
        F0 = F - F.mean(axis=1, keepdims=True)
        Fn = np.linalg.norm(F0, axis=1) + eps

        # adjacent row correlations
        corr = np.sum(F0[:-1] * F0[1:], axis=1) / (Fn[:-1] * Fn[1:])

        scores[t] = np.min(corr)  # tearing → very low min

    return scores


def gradient_energy_spike_score(frames):
    """
    frames: (T, H, W)
    returns: per-frame max vertical gradient energy (T,)
    """
    # vertical gradient
    Gy = np.abs(frames[:, 1:, :] - frames[:, :-1, :])

    # mean over width → (T, H-1)
    row_energy = Gy.mean(axis=2)

    # tearing → one row dominates
    score = np.max(row_energy, axis=1) / (np.median(row_energy, axis=1) + 1e-8)
    return score

