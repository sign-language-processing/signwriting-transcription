# Example Dataset

Used for overfitting and testing purposes, the dataset includes 8 pose from https://whatsthatsign.com/
Specifically, the signs for the numbers 80 to 87.
Numbers are duplicated for the training set, and split between validation and test sets.

## Overfitting Test

With the variables:
- `DATA_DIR`: "example/data"
- `POSES_DIR`: "example/poses"
- `OUTPUT_DIR`: "example/results"
- `DATA_CSV`: "example/data.csv"

We can run the following commands to prepare the data and train a model:
```
make train-overfit
```