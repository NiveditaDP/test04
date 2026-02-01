#Linear regression
#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


#loading the dataset
df = pd.read_csv("../dataset/adult.csv")
print("Linear Regression")
print("\n")
print("First 5 rows of dataset:\n", df.head())
print("\nColumns in dataset:\n", df.columns)


#handling the missing values
#replace " ?" or empty strings with nan
df.replace(" ?", np.nan, inplace=True)
df.dropna(inplace=True)
print("\nMissing values per column:\n", df.isnull().sum())


#checking and crrect data types
#converting the target 'income' to numeric if it exists
if 'income' in df.columns:
    df['income'] = df['income'].map({"<=50K": 0, ">50K": 1}).astype(int)


#eliminatin duplicate records
df.drop_duplicates(inplace=True)


#finding & treating outliers(for numeric columns)
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]


#transforming skewed numeric columns safely
#only transform columns that exist and are numeric
skewed_cols = [col for col in numeric_cols if col in df.columns and col not in ['income']]
for col in skewed_cols:
    df[col] = np.log1p(df[col])


#encoding categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols)
df_encoded = df_encoded.astype(int)

print("\nData after encoding (first 5 rows):")
print(df_encoded.head().to_string(index=False))


#feature selection (drop the columuns that are irrelevant)
if 'education-num' in df_encoded.columns:
    df_encoded.drop(columns=['education-num'], inplace=True)


#feature scaling
X = df_encoded.drop('income', axis=1)
y = df_encoded['income']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


#train linear regression
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


#regression evaluation
def evaluate_regression(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} Regression Metrics:")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

evaluate_regression(y_train, y_train_pred, "Training Set")
evaluate_regression(y_test, y_test_pred, "Test Set")


#classification metrics
y_test_class = (y_test_pred >= 0.5).astype(int)
print("\nClassification Metrics:")
print("Accuracy :", accuracy_score(y_test, y_test_class))
print("Precision:", precision_score(y_test, y_test_class))
print("Recall   :", recall_score(y_test, y_test_class))
print("F1-Score :", f1_score(y_test, y_test_class))
print("\n")









#decision tree 
#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


#loading the dataset
df = pd.read_csv("../dataset/adult.csv")
print("Decision Tree")
print("\n")
print("First 5 rows of dataset:\n", df.head())
print("\nColumns in dataset:\n", df.columns)


#handling missing values
df.replace(" ?", np.nan, inplace=True)
df.dropna(inplace=True)
print("\nMissing values per column:\n", df.isnull().sum())


#correcting data types
if 'income' in df.columns:
    df['income'] = df['income'].map({"<=50K": 0, ">50K": 1}).astype(int)


#eliminating duplicates
df.drop_duplicates(inplace=True)


#finding and treating outliers (numeric columns)
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]


#transform skewed numeric columns (if there are any means)
skewed_cols = [col for col in numeric_cols if col in df.columns and col != 'income']
for col in skewed_cols:
    df[col] = np.log1p(df[col])


#encoding categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols)
df_encoded = df_encoded.astype(int)

#drop the features if they are irrelevant
if 'education-num' in df_encoded.columns:
    df_encoded.drop(columns=['education-num'], inplace=True)

print("\nData after encoding (first 5 rows):")
print(df_encoded.head().to_string(index=False))


#feature scaling
X = df_encoded.drop('income', axis=1)
y = df_encoded['income']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


#Decision Tree Regression (without hyperparameter tuning)
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)

def evaluate_model(y_true, y_pred, dataset_name="Dataset"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{dataset_name} Metrics:")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

evaluate_model(y_train, y_train_pred, "Training Set (DT)")
evaluate_model(y_test, y_test_pred, "Test Set (DT)")


#Decision Tree Regression (with hyperparameter tuning)
param_grid = {
    'max_depth': [None] + list(np.arange(1, 21)),
    'min_samples_split': np.arange(2, 21),
    'min_samples_leaf': np.arange(1, 21)
}

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_tree = grid.best_estimator_
y_train_pred_best = best_tree.predict(X_train)
y_test_pred_best = best_tree.predict(X_test)

print("\nBest hyperparameters found:", grid.best_params_)
evaluate_model(y_train, y_train_pred_best, "Training Set (Best DT)")
evaluate_model(y_test, y_test_pred_best, "Test Set (Best DT)")


#visualization of the best Decision Tree
plt.figure(figsize=(20,10))
plot_tree(
    best_tree,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.show()
print("\n")









#random forest
#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("Random Forest")
print("\n")

#loading dataset
df = pd.read_csv("../dataset/adult.csv")

#handling missing values
df.replace(" ?", np.nan, inplace=True)
df.dropna(inplace=True)

#fixing target column
df['income'] = df['income'].map({"<=50K": 0, ">50K": 1}).astype(int)

#eliminating duplicates
df.drop_duplicates(inplace=True)

#outlier treatment (numeric columns only, exclude target)
numeric_cols = df.select_dtypes(include=np.number).columns
numeric_cols = numeric_cols.drop('income')

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

#encoding categorical variables
categorical_cols = df.select_dtypes(include='object').columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#feature split
X = df_encoded.drop('income', axis=1)
y = df_encoded['income']

#feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

#random forest classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

rf_model.fit(X_train, y_train)

#predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

#evaluation – training
print("\nTraining Set (Random Forest) Metrics:")
print("Accuracy :", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred))
print("Recall   :", recall_score(y_train, y_train_pred))
print("F1-Score :", f1_score(y_train, y_train_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))

#evaluation – testing
print("\nTest Set (Raandom Forest) Metrics:")
print("Accuracy :", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall   :", recall_score(y_test, y_test_pred))
print("F1-Score :", f1_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("\n")









#k nearest neighbor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#load dataset (change path if needed)
df = pd.read_csv("../dataset/adult.csv")  # Adjust the path if your file is in another folder

print("K Nearest Neighbor")
print("\n")
print("Initial dataset shape:", df.shape)
print("First 5 rows:\n", df.head())


#data preprocessing
#handling missing values: replace ' ?' with np.nan and drop
df.replace(' ?', np.nan, inplace=True)
df.dropna(inplace=True)

#remove duplicates
df.drop_duplicates(inplace=True)

#fix data types: encode target variable 'income'
df['income'] = df['income'].map({"<=50K": "Low", ">50K": "High"})

#identify categorical columns 
categorical_cols = df.select_dtypes(include='string').columns.tolist()

#drop target from categorical list
if 'income' in categorical_cols:
    categorical_cols.remove('income')

#one-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


#features and target
X = df_encoded.drop('income', axis=1)
y = df_encoded['income']


#feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#train-test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


#knn model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

#predictions
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)


#evaluation
print("\nK NEAREST NEIGHBORS RESULTS\n")

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy :", accuracy_score(y_test, y_test_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_test_pred, digits=4))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_test_pred))
print("\n")









#support vector machine 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#load dataset
df = pd.read_csv("../dataset/adult.csv")

print("Support Vector Machine")
print("\n")
print("\nInitial dataset shape:", df.shape)


#data preprocessing

#handling missing values
df.replace(' ?', np.nan, inplace=True)
df.dropna(inplace=True)

#eliminate duplicates
df.drop_duplicates(inplace=True)

#encode target variable 'income'
df['income'] = df['income'].map({"<=50K": "Low", ">50K": "High"})

#identify categorical columns
categorical_cols = df.select_dtypes(include='string').columns.tolist()

#removing target column
if 'income' in categorical_cols:
    categorical_cols.remove('income')

#one-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


#features and target
X = df_encoded.drop('income', axis=1)
y = df_encoded['income']


#feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


#svm model
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

#predictions
y_train_pred = svm.predict(X_train)
y_test_pred = svm.predict(X_test)


#evaluation
print("\nSUPPORT VECTOR MACHINE RESULTS\n")

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy :", accuracy_score(y_test, y_test_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_test_pred, digits=4))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_test_pred))
