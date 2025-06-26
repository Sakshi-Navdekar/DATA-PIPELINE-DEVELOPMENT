import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# EXTRACT
file_path = 'data.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError("Missing 'data.csv'. Put it in the same folder.")

data = pd.read_csv(file_path)
print("Original Data:\n", data)

# TRANSFORM
numerical_features = ['Age', 'Salary']
categorical_features = ['Gender', 'Department']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_features),
    ('cat', cat_pipeline, categorical_features)
])

processed_array = preprocessor.fit_transform(data)
cat_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
columns = numerical_features + list(cat_cols)

processed_df = pd.DataFrame(
    processed_array.toarray() if hasattr(processed_array, 'toarray') else processed_array,
    columns=columns
)

print("\nProcessed Data:\n", processed_df)

# LOAD
processed_df.to_csv('processed_data.csv', index=False)
print("\n✅ Processed data saved as 'processed_data.csv'")
