import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def decode_bytes(df):
    for column in df.columns:
        if df[column].dtype == object:  # Assuming all 'object' types could be strings
            df[column] = df[column].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    return df

def preprocess_and_evaluate(file_path):
    # Load the dataset
    data, meta = arff.loadarff(file_path)
    df_raw = pd.DataFrame(data)
    df_raw = decode_bytes(df_raw)

    # Encode target variable if it's not numeric
    y_raw = pd.Categorical(df_raw['binaryClass']).codes

    # Process the raw data: Encode categorical features
    X_raw = pd.get_dummies(df_raw.drop(columns='binaryClass'))

    # Create LogisticRegression instance
    classifier = LogisticRegression(random_state=0, max_iter=1000)

    # Impute missing values for the raw data
    imputer = SimpleImputer(strategy='mean')
    X_raw_imputed = imputer.fit_transform(X_raw)

    # Evaluate classifier on raw, imputed data
    np.random.seed(42)
    mean_accuracy_raw = cross_val_score(classifier, X_raw_imputed, y_raw, cv=10).mean()
    print("Mean Accuracy with Raw Data:", mean_accuracy_raw)

    # Data preprocessing steps for processed data
    df_missing = df_raw.copy()
    df_missing = decode_bytes(df_missing)

    for column in ['skin', 'axillar', 'sex', 'histologic-type']:
        mode_value = df_missing[column].mode()[0]
        df_missing[column] = df_missing[column].replace('?', mode_value)

    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan,
                                     categories=[['poorly', 'fairly', 'well']])
    df_missing['degree-of-diffe'] = ordinal_encoder.fit_transform(df_missing[['degree-of-diffe']].astype(str))

    X_processed = pd.get_dummies(df_missing.drop(columns='binaryClass'))
    y_processed = pd.Categorical(df_missing['binaryClass']).codes

    # Impute missing values for the processed data
    X_processed_imputed = imputer.fit_transform(X_processed)

    # Evaluate classifier on processed, imputed data
    mean_accuracy_processed = cross_val_score(classifier, X_processed_imputed, y_processed, cv=10).mean()
    print("Mean Accuracy After Data Processing:", mean_accuracy_processed)

    # Plot distributions
    plot_distributions(df_raw, df_missing)

def plot_distributions(df_raw, df_missing):
    selected_features = ['sex', 'skin', 'axillar', 'histologic-type', 'degree-of-diffe']
    for feature in selected_features:
        plt.figure(figsize=(8, 6))
        plt.ylim(0, 350)

        plt.subplot(1, 2, 1)
        plt.hist(df_raw[feature].dropna(), bins=30, color='blue', alpha=0.7)
        plt.title(f'Distribution of {feature} Before Preprocessing', fontsize=10)
        plt.xlabel(feature, fontsize=8)
        plt.ylabel('Frequency', fontsize=8)
        plt.xticks(rotation=45, fontsize=8)
        plt.ylim(0, 350)

        plt.subplot(1, 2, 2)
        plt.hist(df_missing[feature].dropna(), bins=30, color='green', alpha=0.7)
        plt.title(f'Distribution of {feature} After Preprocessing', fontsize=10)
        plt.xlabel(feature, fontsize=8)
        plt.ylabel('Frequency', fontsize=8)
        plt.xticks(rotation=45, fontsize=8)
        plt.ylim(0, 350)
        plt.tight_layout()
        plt.savefig(f'{feature}_distribution.png', dpi=300)
        plt.close()

# Adjust the file path as needed
preprocess_and_evaluate(r"C:\Users\irani\Downloads\ML Dr. Lars\primary-tumor.arff")


'''''''''
results:
Mean Accuracy with Raw Data: 0.8493761140819964
Mean Accuracy After Data Processing: 0.8524064171122994
'''