from pathlib import Path
import cv2
import pandas as pd

# =========================
# USER SETTINGS
# Update ROOT_BASE if running on a different computer or OS.
# =========================

ROOT_BASE = Path(r"\\filenest.dyn.berkeley.edu\Wilbrecht_file_server\HongliWang\Openfield_ASD_redlight")

STRAIN = "Scn2A"
# STRAIN = "Syngap"

ROOT = ROOT_BASE / STRAIN
SUMMARY = ROOT / "summary"
PREFLIGHT = SUMMARY / "redlight_preflight.csv"

OVERWRITE = False


# =========================
# Helper functions
# =========================

def mouse_callback(event, x, y, flags, param):
    points, frame_disp = param

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(frame_disp, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(
                frame_disp,
                str(len(points)),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


def annotate_video(video_path, output_csv):
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Could not read first frame: {video_path}")
        return False

    points = []
    frame_disp = frame.copy()

    window_name = "Click arena corners: UL, UR, LR, LL | s=save, r=reset, q=skip"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 800)
    cv2.setMouseCallback(window_name, mouse_callback, [points, frame_disp])

    while True:
        show = frame_disp.copy()

        instructions = [
            "Click true floor corners in order:",
            "1 upper-left, 2 upper-right, 3 lower-right, 4 lower-left",
            "s = save, r = reset, q = skip"
        ]

        y0 = 30
        for line in instructions:
            cv2.putText(
                show,
                line,
                (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            y0 += 28

        cv2.imshow(window_name, show)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            points.clear()
            frame_disp = frame.copy()
            cv2.setMouseCallback(window_name, mouse_callback, [points, frame_disp])
            print("Reset points.")

        elif key == ord("s"):
            if len(points) != 4:
                print("Need exactly 4 points before saving.")
                continue

            output_csv.parent.mkdir(parents=True, exist_ok=True)

            arena = {
                "upper left": [points[0]],
                "upper right": [points[1]],
                "lower right": [points[2]],
                "lower left": [points[3]],
            }

            pd.DataFrame(arena).to_csv(output_csv, index=False)

            print("Saved:", output_csv)
            cv2.destroyWindow(window_name)
            return True

        elif key == ord("q"):
            print("Skipped:", video_path)
            cv2.destroyWindow(window_name)
            return False


# =========================
# Main
# =========================

if not PREFLIGHT.exists():
    raise FileNotFoundError(f"Preflight file not found: {PREFLIGHT}")

df = pd.read_csv(PREFLIGHT)

# 只标注 AnimalList 里 genotype 有效的 session
df = df[df["validGenotype"] == True].copy()

print(f"Valid genotype sessions to check: {len(df)}")

for _, row in df.iterrows():
    obsID = str(row["obsID"])
    video_path = Path(row["videoPath"])
    output_csv = SUMMARY / obsID / "arena_coordinates.csv"

    if output_csv.exists() and not OVERWRITE:
        print("Already exists, skip:", output_csv)
        continue

    print("\n====================================")
    print("Animal:", row["animalID"])
    print("obsID:", obsID)
    print("Age:", row["ageDays"], row["ageGroup"])
    print("Genotype:", row["genotype"])
    print("Video:", video_path)
    print("Output:", output_csv)
    print("====================================")

    annotate_video(video_path, output_csv)

print("Done.")
