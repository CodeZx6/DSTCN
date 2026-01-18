# Data Directory

Place your preprocessed data files in this directory:

## Required Files

- `raw_edge80_in_day.npy`: OD inflow matrix (T, 80, 80)
- `raw_edge80_out_day.npy`: OD outflow matrix (T, 80, 80)
- `time_Matrix_regularization.npy`: Travel time matrix (80, 80)
- `raw_od_out_distribution.npy`: OD outflow distribution (T, 80, 80)
- `raw_edge80_in_matrix{1-6}.npy`: Completed OD matrices with delays
- `raw_flow80_in_day_delay{1-6}_version_expand.npy`: Delayed flow expansions
- `raw_od_distribution{1-6}.npy`: OD distribution probability matrices

## Data Format

- T: Number of time steps
- 80: Number of metro stations
- All matrices should be numpy arrays (.npy format)

## Notes

- Data files are not included in version control (.gitignore)
- Ensure data is properly preprocessed before training
- Contact authors for data preprocessing scripts
