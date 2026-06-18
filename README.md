# mouscope
# a complete data analysis package for behavioral and calcium imaging analysis
# 1. openfield
# 2. rotarod
# 3. odor discrimination
# 4. 4-choice digging


## 1. Data structure

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
