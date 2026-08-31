from behavioral_pipeline import *
from pathlib import Path
import os
import re
import numpy as np
import pandas as pd

# =========================
# USER SETTINGS
# Update ROOT_BASE if running on a different computer or OS.
# =========================

STRAIN = "Scn2A"
# STRAIN = "Syngap"

ROOT_BASE = Path(r"\\filenest.dyn.berkeley.edu\Wilbrecht_file_server\HongliWang\Openfield_ASD_redlight")
ROOT = ROOT_BASE / STRAIN
SUMMARY = ROOT / "summary"
PREFLIGHT = SUMMARY / "redlight_preflight.csv"

RUN_MOTION = True
RUN_CENTER = True


def valid_genotype(x):
    x = str(x).strip().upper()
    return x in ["WT", "HET"]


print("Loading:", ROOT)
OF = BehDataOF(str(ROOT), STRAIN)

pre = pd.read_csv(PREFLIGHT)
pre["animalID"] = pre["animalID"].astype(str)
pre["obsID"] = pre["obsID"].astype(str)

valid_pre = pre[pre["validGenotype"] == True].copy()

valid_obsIDs = valid_pre["obsID"].tolist()
valid_animals = valid_pre["animalID"].tolist()

# Build obsID for each row in OF.data_index
obsIDs = []
animalIDs = []

for idx, row in OF.data_index.iterrows():
    animal = str(row["Animal"])
    obj = row["DLC_obj"]
    session_folder = Path(str(obj.videoPath)).parent.name
    obsID = f"{animal}_{session_folder}"

    animalIDs.append(animal)
    obsIDs.append(obsID)

OF.data_index = OF.data_index.copy()
OF.data_index["animalID_raw"] = animalIDs
OF.data_index["obsID"] = obsIDs

keep = OF.data_index["obsID"].isin(valid_obsIDs)
OF.data_index = OF.data_index.loc[keep].reset_index(drop=True)

# Use obsID as analysis identity to avoid session overwriting
OF.Animals = OF.data_index["obsID"].astype(str).values
OF.nSubjects = len(OF.Animals)

# Recompute grouping indices
genotype = (
    OF.data_index["Genotype"]
    .astype(str)
    .str.strip()
    .str.upper()
    .values
)

animalIdx = np.arange(OF.nSubjects)

OF.WTIdx = animalIdx[genotype == "WT"]
OF.MutIdx = animalIdx[genotype == "HET"]

OF.Gender = OF.data_index["Gender"].values
OF.maleIdx = np.where(OF.data_index["Gender"] == "M")[0]
OF.femaleIdx = np.where(OF.data_index["Gender"] == "F")[0]

print("\nIncluded sessions:", OF.nSubjects)
print("Animals / obsIDs:")
for obs in OF.Animals:
    print(" ", obs)

print("\nWTIdx:", OF.WTIdx)
print("MutIdx:", OF.MutIdx)
print("maleIdx:", OF.maleIdx)
print("femaleIdx:", OF.femaleIdx)

SUMMARY.mkdir(parents=True, exist_ok=True)

if RUN_MOTION:
    print("\nRunning motion_analysis...")
    OF.motion_analysis(str(SUMMARY))

if RUN_CENTER:
    print("\nRunning center_analysis...")
    OF.center_analysis(str(SUMMARY))

print("\nDone.")
print("Summary folder:", SUMMARY)
