"""Copy all AVI recordings below the rotarod data folder for LITpose."""

from pathlib import Path
from shutil import copy2


SOURCE_FOLDER = Path(r"Y:\HongliWang\Rotarod\ASD_strains\TSC2_adol\Data")
DESTINATION_FOLDER = Path(
    r"Y:\HongliWang\Rotarod\ASD_strains\TSC2_adol\video4litpose"
)


def copy_avi_videos(source: Path = SOURCE_FOLDER, destination: Path = DESTINATION_FOLDER) -> int:
    """Copy every AVI below *source* into *destination*.

    The destination is deliberately flat for LITpose.  A duplicate filename is
    reported instead of overwriting an existing video from another subfolder.
    """
    if not source.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {source}")

    destination.mkdir(parents=True, exist_ok=True)
    videos = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() == ".avi")

    copied = 0
    for video in videos:
        target = destination / video.name
        if target.exists():
            raise FileExistsError(
                f"Not copying {video}: {target} already exists. "
                "Rename one of the videos before retrying."
            )
        copy2(video, target)
        copied += 1
        print(f"Copied: {video} -> {target}")

    print(f"Copied {copied} AVI video(s) to {destination}")
    return copied


if __name__ == "__main__":
    copy_avi_videos()
