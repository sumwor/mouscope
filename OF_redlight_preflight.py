from behavioral_pipeline import *
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# USER SETTINGS
# Update ROOT_BASE if running on a different computer or OS.
# =========================

ROOT_BASE = Path(r"\\filenest.dyn.berkeley.edu\Wilbrecht_file_server\HongliWang\Openfield_ASD_redlight")

STRAIN = "Scn2A"
# STRAIN = "Syngap"

ROOT = ROOT_BASE / STRAIN
SUMMARY = ROOT / "summary"
SUMMARY.mkdir(parents=True, exist_ok=True)


def parse_age_from_text(text):
    m = re.findall(r"(?i)(?:^|[^A-Za-z0-9])p(\d{1,3})(?=[^A-Za-z0-9]|$)", str(text))
    return int(m[0]) if m else np.nan

def parse_age_group(age):
    if pd.isna(age):
        return "Other"

    age = int(age)

    if age == 15:
        return "P15"

    elif 29 <= age <= 35:
        return "P30"

    elif age >= 60:
        return "P60+"

    else:
        return "Other"


def valid_genotype(x):
    x = str(x).strip().upper()
    return x in ["WT", "HET"]


print("ROOT_BASE:", ROOT_BASE)
print("ROOT:", ROOT)

if not ROOT.exists():
    raise FileNotFoundError(f"Strain root does not exist: {ROOT}")

print("Loading BehDataOF...")
OF = BehDataOF(str(ROOT), STRAIN)

rows = []

for idx, row in OF.data_index.iterrows():
    animal = str(row["Animal"])
    genotype = str(row["Genotype"]).strip()
    sex = str(row["Gender"]).strip()

    obj = row["DLC_obj"]
    video_path = str(obj.videoPath)
    session_folder = Path(video_path).parent.name

    ageDays = parse_age_from_text(video_path)
    ageGroup = parse_age_group(ageDays)

    # Get session-level observation ID
    obsID = f"{animal}_{session_folder}"

    # The current center_analysis reads arena coordinates from: summary/animalID/arena_coordinates.csv
    # If we switch to session-level arena annotations later, this should be changed to summary/obsID/arena_coordinates.csv
    arena_animal_csv = SUMMARY / animal / "arena_coordinates.csv"
    arena_obs_csv = SUMMARY / obsID / "arena_coordinates.csv"

    rows.append({
        "rowIndex": idx,
        "animalID": animal,
        "obsID": obsID,
        "genotype": genotype,
        "sex": sex,
        "validGenotype": valid_genotype(genotype),
        "ageDays": ageDays,
        "ageGroup": ageGroup,
        "sessionFolder": session_folder,
        "arenaAnimalExists": arena_animal_csv.exists(),
        "arenaObsExists": arena_obs_csv.exists(),
        "videoPath": video_path
    })

df = pd.DataFrame(rows)

out = SUMMARY / "redlight_preflight.csv"
df.to_csv(out, index=False)

print("\nSaved:")
print(out)

print("\nTotal sessions:", len(df))
print("Valid genotype sessions:", int(df["validGenotype"].sum()))
print("Excluded no-genotype sessions:", int((~df["validGenotype"]).sum()))

valid_df = df[df["validGenotype"]].copy()

# =========================
# Age x Genotype x Sex sample size table
# =========================

print("\nAge group x genotype x sex among valid sessions:")

sample_size_table = pd.crosstab(
    [valid_df["ageGroup"], valid_df["genotype"]],
    valid_df["sex"]
)

# Add total N for each Age x Genotype group
sample_size_table["Total"] = sample_size_table.sum(axis=1)

print(sample_size_table)

sample_size_out = SUMMARY / "age_genotype_sex_sample_sizes.csv"

sample_size_table.to_csv(
    sample_size_out
)

print("\nSaved sample-size table:")
print(sample_size_out)

# ============================================================
# Quality control
# ============================================================

print("\nRepeated animalIDs among valid sessions:")
dup = valid_df[valid_df["animalID"].duplicated(keep=False)].sort_values(["animalID", "ageDays"])
print(dup[["animalID", "obsID", "genotype", "sex", "ageDays", "ageGroup", "sessionFolder"]])

print("\nMissing age among valid sessions:")
print(valid_df[valid_df["ageGroup"] == "Other"][["animalID", "obsID", "videoPath"]])

print("\nMissing session-level arena annotations among valid sessions:")
print(
    valid_df[
        ~valid_df["arenaObsExists"]
    ][
        ["animalID", "obsID", "videoPath"]
    ]
)

print("\nExcluded because genotype missing:")
print(df[~df["validGenotype"]][["animalID", "obsID", "genotype", "videoPath"]])
