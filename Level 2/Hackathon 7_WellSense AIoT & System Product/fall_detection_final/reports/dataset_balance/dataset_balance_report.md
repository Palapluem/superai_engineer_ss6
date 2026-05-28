# Dataset Balance And Missingness Report

This report explains whether the received dataset is imbalanced and where class/missing-value issues come from.

## Short Answer

- Yes, `windows_all.csv` is imbalanced.
- No, `windows_all.csv` does not have only one class. It has 11 classes in `class_en`.
- The file that really has only one class is `joined_imu_windows.csv` (`slow_collapse_fall` only).
- The confusing missing values are mostly from the `class` column, but `class_en` is complete and should be used as the canonical class label.

## windows_all.csv

- Rows: 2138
- `class_en` unique classes: 11
- Class imbalance ratio max/min: 89.6:1
- Largest class: `sideways_fall` = 806
- Smallest class: `standing` = 9

Category counts:

| category        |   count |
|:----------------|--------:|
| fall            |    1825 |
| activity        |     287 |
| static_activity |      26 |

Class counts:

| class_en               |   count |
|:-----------------------|--------:|
| sideways_fall          |     806 |
| backward_fall          |     805 |
| slow_collapse_fall     |     189 |
| limping_walk           |      80 |
| elderly_pick_up_object |      63 |
| corrected_walking      |      59 |
| stand_sit_alternating  |      56 |
| normal_walk            |      29 |
| gradual_fall           |      25 |
| lying_down             |      17 |
| standing               |       9 |

Label/missing columns:

| column   |   missing |   present |
|:---------|----------:|----------:|
| class_en |         0 |      2138 |
| class    |      1800 |       338 |
| class_th |       338 |      1800 |
| category |         0 |      2138 |
| label    |         0 |      2138 |

## File-Level Summary

| file                    | column   | available   |   rows |   unique_non_null |   null_count |   max_min_ratio | top_class          |   top_class_percent |   is_single_class |
|:------------------------|:---------|:------------|-------:|------------------:|-------------:|----------------:|:-------------------|--------------------:|------------------:|
| windows_all.csv         | class_en | True        |   2138 |                11 |            0 |        89.5556  | sideways_fall      |            37.6988  |                 0 |
| windows_all.csv         | class    | True        |   2138 |                 8 |         1800 |         8.88889 | limping_walk       |             3.74181 |                 0 |
| windows_all.csv         | category | True        |   2138 |                 3 |            0 |        70.1923  | fall               |            85.3601  |                 0 |
| windows_all.csv         | label    | True        |   2138 |                 2 |            0 |         5.83067 | 1                  |            85.3601  |                 0 |
| windows_extracted.csv   | class_en | True        |    338 |                 8 |            0 |         8.88889 | limping_walk       |            23.6686  |                 0 |
| windows_extracted.csv   | class    | True        |    338 |                 8 |            0 |         8.88889 | limping_walk       |            23.6686  |                 0 |
| windows_extracted.csv   | category | True        |    338 |                 3 |            0 |        11.48    | activity           |            84.9112  |                 0 |
| windows_extracted.csv   | label    | True        |    338 |                 2 |            0 |        12.52    | 0                  |            92.6036  |                 0 |
| joined_imu_windows.csv  | class_en | True        |   1800 |                 1 |            0 |         1       | slow_collapse_fall |           100       |                 1 |
| joined_imu_windows.csv  | class    | False       |    nan |               nan |          nan |       nan       | nan                |           nan       |               nan |
| joined_imu_windows.csv  | category | True        |   1800 |                 1 |            0 |         1       | fall               |           100       |                 1 |
| joined_imu_windows.csv  | label    | False       |    nan |               nan |          nan |       nan       | nan                |           nan       |               nan |
| merged_dataset_full.csv | class_en | True        |   9860 |                11 |            0 |         8.03448 | limping_walk       |            16.5416  |                 0 |
| merged_dataset_full.csv | class    | True        |   9860 |                 3 |         8060 |         4.26455 | ล้มข้าง              |             8.17444 |                 0 |
| merged_dataset_full.csv | category | True        |   9860 |                 3 |            0 |        10.2787  | activity           |            59.8377  |                 0 |
| merged_dataset_full.csv | label    | True        |   9860 |                 1 |         8060 |         1       | 1.0                |            18.2556  |                 1 |

## Recommended Use

- Use `windows_all.csv` for Model 2 feature-window training/testing because it has the broadest class coverage.
- Use `class_en`, `category`, and `label`; avoid using `class` as the main class label because it is missing for many rows.
- Treat `joined_imu_windows.csv` as a sequence/hybrid demo file only, because it contains only `slow_collapse_fall`.
- Treat `merged_dataset_full.csv` carefully because many columns are sparse due to merging raw rows and window rows with different schemas.

## Generated Graphs

- `windows_all_class_distribution.png`
- `windows_all_category_distribution.png`
- `class_distribution_across_files.png`
- `windows_all_label_column_missingness.png`
- `top_missing_columns.png`