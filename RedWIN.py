import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer

file_path = r"C:\Users\irani\Downloads\ML Dr. Lars\winequality-red.csv"
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

clfr_no_preprocessing = RandomForestClassifier(n_estimators=100, random_state=0)


clfr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', clfr_no_preprocessing)
])

cross_val_scores_no_preprocessing = cross_val_score(clfr_no_preprocessing, X_red, y_red, cv=10)
print("RF without preprocessing mean score: {:.4f}".format(cross_val_scores_no_preprocessing.mean()))
cross_val_scores = cross_val_score(clfr_pipeline, X_red, y_red, cv=10)
print("RF with preprocessing mean score: {:.4f}".format(cross_val_scores.mean()))

#plot
fig, axs = plt.subplots(2, len(features), figsize=(20, 8))

for i, feature in enumerate(features):
    axs[0, i].hist(X_red[feature], bins=20, color='blue', alpha=0.5, label='Before')
    axs[1, i].hist(X_red[feature].apply(np.log1p), bins=20, color='orange', alpha=0.5, label='After')
    axs[0, i].set_title(feature, fontsize=10)
    axs[1, i].set_title(feature, fontsize=10)
    axs[0, i].legend(fontsize=10)
    axs[1, i].legend(fontsize=10)

axs[0, 0].set_ylabel('Frequency', fontsize=10)
axs[1, 0].set_ylabel('Frequency', fontsize=10)
plt.suptitle('Feature Distributions Before and After Preprocessing', fontsize=12)
plt.tight_layout()
plt.show()


''''
#results:
RF without preprocessing mean score: 0.5710
RF with preprocessing mean score: 0.5829
'''