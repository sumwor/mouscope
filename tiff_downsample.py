import numpy as np
import tifffile as tiff
from pathlib import Path

# ------------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------------
in_tiff = Path(r"Y:\HongliWang\Miniscope\ASDC001\ASDC001_260113\caiman_result\background_movie_160113.tif")
out_tiff = Path(r"Y:\HongliWang\Miniscope\ASDC001\ASDC001_260113\caiman_result\background_movie_ds5_160113.tif")

ds_factor = 5          # temporal downsampling factor
dtype_out = np.uint16 # change if needed (uint16 is typical)

# ------------------------------------------------------------------
# STREAMING TEMPORAL DOWNSAMPLING
# ------------------------------------------------------------------
with tiff.TiffFile(in_tiff) as tif:
    n_frames = len(tif.pages)
    print(f"Total frames: {n_frames}")

    with tiff.TiffWriter(out_tiff, bigtiff=True) as writer:
        buffer = []

        for i, page in enumerate(tif.pages):
            frame = page.asarray()
            buffer.append(frame.astype(np.float32))

            # Once we have ds_factor frames → average & write
            if len(buffer) == ds_factor:
                avg_frame = np.mean(buffer, axis=0)
                writer.write(avg_frame.astype(dtype_out))
                buffer.clear()

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} / {n_frames} frames")

        # Optional: handle leftover frames
        if buffer:
            avg_frame = np.mean(buffer, axis=0)
            writer.write(avg_frame.astype(dtype_out))

print("Done.")
print(f"Saved to: {out_tiff}")