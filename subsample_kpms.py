# subsample DLC/Videos for keypoint-moseq
# currently the odor videos are too large to process

# subsample 5 mins from beginning/middle/end of every session
import glob
import os
import cv2
import pandas as pd
from tqdm import tqdm 

root_dir = r'Y:\HongliWang\Juvi_ASD Deterministic\TSC2_withRec'
video_dir = r'Y:\HongliWang\Juvi_ASD Deterministic\TSC2_withRec\videos'

video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
dlc_files = glob.glob(os.path.join(video_dir, '*.csv'))
video_by_stem = {os.path.splitext(os.path.basename(v))[0]: v for v in video_files}

kpms_dir = os.path.join(root_dir, 'video_kpms')
if not os.path.exists(kpms_dir):
    os.makedirs(kpms_dir)


def actual_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video for frame counting: {video_path}')

    count = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        count += 1
    cap.release()
    return count


def save_video_segment_by_frame(video_path, out_path, start_frame, end_frame, fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video for reading: {video_path}')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Could not open video writer for: {out_path}')

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            writer.release()
            raise RuntimeError(
                f'Could not read frame {frame_idx} from {video_path}. '
                f'Read {frame_idx - start_frame} frames before failure.'
            )
        writer.write(frame)

    cap.release()
    writer.release()
    return end_frame - start_frame

for dlc in tqdm(dlc_files):
    dlc_header = pd.read_csv(dlc, header=None, nrows=3)
    dlc_data = pd.read_csv(dlc, header=None, skiprows=3)
    nFrames = len(dlc_data)

    dlc_stem = os.path.splitext(os.path.basename(dlc))[0][0:27]
    video = video_by_stem.get(dlc_stem)
    if video is None:
        matches = [v for stem, v in video_by_stem.items() if stem in dlc_stem or dlc_stem in stem]
        if not matches:
            raise FileNotFoundError(f'No matching video found for {dlc}')
        video = matches[0]

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        raise ValueError(f'Could not read FPS from {video}')
    if video_nframes <= 0:
        video_nframes = actual_frame_count(video)
    elif abs(video_nframes - nFrames) > 0:
        actual_nframes = actual_frame_count(video)
        if actual_nframes != video_nframes:
            print(
                f'Warning: {video} reported {video_nframes} frames via CAP_PROP_FRAME_COUNT '
                f'but actual counted {actual_nframes} frames.'
            )
            video_nframes = actual_nframes

    if video_nframes != nFrames:
        raise ValueError(
            f'Frame count mismatch for {video}: video has {video_nframes} frames but DLC csv has {nFrames} rows'
        )

    video_stem = os.path.splitext(os.path.basename(video))[0]

    for label, start_pct, end_pct in [('05-15', 0.05, 0.15),
                                      ('50-60', 0.50, 0.60),
                                      ('85-95', 0.85, 0.95)]:
        start_frame = int(nFrames * start_pct)
        end_frame = int(nFrames * end_pct)

        dlc_segment = dlc_data.iloc[start_frame:end_frame].copy()
        dlc_segment.iloc[:, 0] = range(len(dlc_segment))
        pd.concat([dlc_header, dlc_segment], ignore_index=True).to_csv(
            os.path.join(kpms_dir, f'{dlc_stem}_{label}.csv'),
            header=False,
            index=False
        )

        out_video = os.path.join(kpms_dir, f'{video_stem}_{label}.mp4')
        expected_nframes = len(dlc_segment)
        actual_nframes = save_video_segment_by_frame(video, out_video, start_frame, end_frame, fps)
        if actual_nframes != expected_nframes:
            raise RuntimeError(
                f'Video segment frame count mismatch for {out_video}: '
                f'expected {expected_nframes}, got {actual_nframes}'
            )
