# downsample cnmfe results to 1:10 on time 
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import caiman as cm


#root_dir = r''

filepath = r'Y:\HongliWang\Miniscope\ASD\Data\ASDC001\Odor\Imaging\ASDC001_20260113\caiman_results\ASDC001_AB_ImgVideo_2026-01-13_caiman_result_ds.hdf5'

cnmfe_model = load_CNMF(filepath)

ds10_cnmfe_model =cnmfe_model

ds10_cnmfe_model.estimates.C = ds10_cnmfe_model.estimates.C[:,::10]
ds10_cnmfe_model.estimates.S = ds10_cnmfe_model.estimates.S[:,::10]
ds10_cnmfe_model.estimates.F_dff = ds10_cnmfe_model.estimates.F_dff[:,::10]
save_path =  r'Y:\HongliWang\Miniscope\ASD\Data\ASDC001\Odor\Imaging\ASDC001_20260113\caiman_results\ASDC001_AB_ImgVideo_2026-01-13_caiman_result_ds10.hdf5'  # or add full/path/to/file.hdf5 # squirrel away correlation image with cnmf object
ds10_cnmfe_model.save(save_path)

# 106897 - 107045

# load mmap file and delete these frames 106897 - 107045
# import os
# import glob
# import tifffile as tiff
# import numpy as np
# import tqdm
# os.environ['CAIMAN_DATA']= r'Y:\HongliWang\Miniscope\ASD\Data\ASDC001\Odor\Imaging\ASDC001_20260112\temp'
# tiff_path = r'Y:\HongliWang\Miniscope\ASD\Data\ASDC001\Odor\Imaging\ASDC001_20260112\motion_corrected_tiffs'
# tiff_files = glob.glob(os.path.join(tiff_path, '*.tif')) + \
#         glob.glob(os.path.join(tiff_path, '*.tiff'))

# # go over tiff files, remove the frames 106897 - 107045, and save as memmap

# remove_start = 106897
# remove_end   = 107045

# global_idx = 0

# for f in tqdm.tqdm(tiff_files):
#   # (frames, H, W)

#     n_frames = 1000
#     file_start = global_idx
#     file_end   = global_idx + n_frames - 1

#     # only act if overlap exists
#     if not (remove_end < file_start or remove_start > file_end):
#         print(f)
#         with tiff.TiffFile(f) as tif:
#             data = tif.asarray() 
#         # compute local indices
#         local_start = max(remove_start, file_start) - file_start
#         local_end   = min(remove_end, file_end)   - file_start

#         print(f"Editing {f}")
#         print(f"  removing frames {local_start} → {local_end}")

#         # remove frames
#         keep_mask = np.ones(n_frames, dtype=bool)
#         keep_mask[local_start:local_end+1] = False
#         new_data = data[keep_mask]

#         # overwrite safely: write temp file then replace
#         tmp_path = f + '.tmp.tif'
#         tiff.imwrite(tmp_path, new_data)

#         os.replace(tmp_path, f)  # atomic replace

#     global_idx += n_frames

# print("Done.")

# #Yr, dims, T = cm.load_memmap(mmap_path)
# fname_DS = cm.save_memmap(tiff_files,base_name='ASDC001_AB_ImgVideo_2026-01-12_ds2_'+str(0)+'px_border_', resize_fact=(0.5,0.5,1), order='C',
#                         border_to_0=0)