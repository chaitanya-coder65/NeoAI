# Step 4 – Data Preprocessing

Run from the project root after completing Step 3:

```bash
python -m src.data_preprocessing
```

Inputs from `artifacts/data/`:
- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`

Outputs:
- `X_train_processed.csv`
- `X_test_processed.csv`
- `y_train_processed.csv`
- `y_test_processed.csv`
- `artifacts/models/preprocessor.pkl`

The preprocessing pipeline uses median imputation followed by standard scaling.
It is fitted only on training features and reused unchanged for test features.
Target labels are preserved without scaling.
