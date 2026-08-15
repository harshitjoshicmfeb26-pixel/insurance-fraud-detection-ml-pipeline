# Dataset

The local reproduction dataset is:

~~~text
data/raw/FraudDataset.csv
~~~

Verified properties:

- Rows: 15,420
- Columns: 33
- Target: FraudFound_P
- Fraud rows: 923
- Final predictors: 29

The 29 predictors are obtained after excluding the target and dropping:

- PolicyNumber
- RepNumber
- Year

This is the dataset used by notebooks/archive/FinalCode123.ipynb and is intended to support local reproduction of the project experiments.

The dataset should not be assumed to be redistributable. Check the original dataset source and license before sharing or publishing it.
