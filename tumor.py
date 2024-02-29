import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt

def preprocess_and_evaluate(file_path):
    data, meta = arff.loadarff(file_path)
    df_raw = pd.DataFrame(data, dtype=str)
    print(df_raw)
    print("Number of instances:", len(df_raw))
    print("Number of symbolic features:", len(df_raw.columns))
    df_missing = df_raw.copy()


    for column in ['skin', 'axillar', 'sex', 'histologic-type']:
        mode_value = df_missing[column].mode()[0]
        df_missing[column] = df_missing[column].replace('?', mode_value)

    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, categories=[[ 'poorly','fairly','well']])
    df_missing['degree-of-diffe'] = ordinal_encoder.fit_transform(df_missing[['degree-of-diffe']])

    X_processed = df_missing.drop(columns='binaryClass')
    y_processed = df_missing['binaryClass']
    X_processed = pd.get_dummies(X_processed)
    classifier = RandomForestClassifier(random_state=0)


    np.random.seed(42)
    mean_accuracy_processed = cross_val_score(classifier, X_processed, y_processed, cv=10).mean()
    print("Mean Accuracy After Data Processing:", mean_accuracy_processed)


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


preprocess_and_evaluate(r"C:\Users\irani\Desktop\primary-tumor.arff")


#results:

'''
       age     sex histologic-type  ... mediastinum abdominal binaryClass
0     >=60  female               ?  ...          no        no           P
1     >=60    male               ?  ...         yes        no           P
2    30-59  female           adeno  ...          no        no           N
3    30-59  female           adeno  ...          no        no           N
4    30-59  female           adeno  ...          no        no           N
..     ...     ...             ...  ...         ...       ...         ...
334  30-59  female           adeno  ...          no        no           N
335  30-59    male      epidermoid  ...          no        no           P
336  30-59  female           adeno  ...          no        no           N
337  30-59  female           adeno  ...          no       yes           N
338  30-59  female           adeno  ...          no        no           N

[339 rows x 18 columns]
Number of instances: 339
Number of symbolic features: 18
Mean Accuracy After Data Processing: 0.8521390374331551
'''
