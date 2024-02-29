import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

file_path = r"C:\Users\irani\Desktop\winequality-red.csv"
df_red = pd.read_csv(file_path, sep=';')
features = df_red.drop('quality', axis=1).columns.tolist()
X_red, y_red = df_red[features], df_red['quality']


def log_transform(X):
    return np.log1p(X)

preprocessor = ColumnTransformer(
    transformers=[
        ('log_transform', FunctionTransformer(log_transform), features),
        ('std_scaler', StandardScaler(), features)
    ]
)

clfr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=0))
])


fig, axs = plt.subplots(2, len(features), figsize=(20, 8))

for i, feature in enumerate(features):
    axs[0, i].hist(X_red[feature], bins=20, color='blue', alpha=0.5, label='Before')
    axs[1, i].hist(log_transform(X_red[feature]), bins=20, color='orange', alpha=0.5, label='After')
    axs[0, i].set_title(feature, fontsize=10)
    axs[1, i].set_title(feature, fontsize=10)
    axs[0, i].legend(fontsize=10)
    axs[1, i].legend(fontsize=10)

axs[0, 0].set_ylabel('Frequency', fontsize=10)
axs[1, 0].set_ylabel('Frequency', fontsize=10)
plt.suptitle('Feature Distributions Before and After Preprocessing', fontsize=12)
plt.tight_layout()
plt.show()


cross_val_scores = cross_val_score(clfr_pipeline, X_red, y_red, cv=10)

print("Log. Reg. with log transformation and StandardScaler mean score: {:.4f}".format(cross_val_scores.mean()))




'''
Log. Reg. with log transformation and StandardScaler mean score: 0.5872

'''