# mouscope
# a complete data analysis package for behavioral and calcium imaging analysis
# 1. openfield
# 2. rotarod
# 3. odor discrimination
# 4. 4-choice digging


## 1. Data structure
to do: currently different behaviors (OF/odor/rotarod) is saved in different root folders, plan to move them to the same root

### odor behavior
The expected folder structure is organized by strain, animal, and data type.

```text
root/
  ├─ strains/
  │   └─ TSC2/
  │       ├─ Data/
  │       │   └─ 111/
  │       │       ├─ odor/
  │       │       │   ├─ Behavior/
  │       │       │   │   └─ ASD111_260101_AB.mat
  │       │       │   └─ BehavioralRecording/
  │       │       │       └─ 260101/
  │       │       │           └─ behavior videos
  │       │       │           └─ DLC .csv file
  │       │       │   └─ Imaging/
  │       │       │       └─ ASDC001_20260112\
  │       │       │           └─ caiman_results\
  │       │       │           └─ motion_corrected_tiffs\
  │       │       │           └─ updated_cnmf.mat
  │       │       └─ ...
  │       ├─ Analysis/
  │       └─ Summary/
  └─ ...
```

- `root/` : top-level project directory
- `strains/` : contains strain folders such as `TSC2`
- `Data/` : raw data organized per animal ID
- `odor/` : odor-specific experiment data
- `Behavior/` : behavioral data files
- `Imaging/` : imaging sessions and video files
- `Analysis/` : analysis outputs
- `Summary/` : summary reports


### rotarod behavior
The expected folder structure is organized by strain, animal, and data type.

```text
root/
  ├─ strains/
  │   └─ TSC2_adol/
  │       ├─ Data/
  │       │   └─ ASD111/
  │       │       ├─ Rotarod/
  │       │       │   ├─ BehavioralRecording/
  │       │       │       └─ ASD111_260101\ 
  │       │       │           └─ASD111_trial12026-01-01T13_01_01.avi
  │       │       │           └─ASD111_trial1_speed2026-01-01T13_01_01.csv 
  │       │       │           └─ASD111_trial1_timeStamp2026-01-01T13_01_01.csv 
  │       │       │           └─ASD111_trial1_speed2026-01-01T13_01_01DLC_*_*_*_filtered.csv 
  │       │       └─ ...
  │       ├─ Analysis/
  │       └─ Summary/
  │       └─ animalList.csv
  │       └─ RR_results.csv
  └─ ...
```

- `root/` : top-level project directory
- `strains/` : contains strain folders such as `TSC2`
- `Data/` : raw data organized per animal ID
- `Rotarod/` : Rotarod-specific experiment data
- `BehavioralRecording/` : behavioral recording videos
- `Imaging/` : imaging sessions and video files
- `Analysis/` : analysis outputs
- `Summary/` : summary reports
- `RR_results.csv` : rotarod performance  
