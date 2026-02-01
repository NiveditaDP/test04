PROJECT TITLE

Income Prediction using Machine Learning Algorithms (Adult Census Dataset)


PROBLEM STATEMENT

The objective of this project is to predict whether an individual’s annual income exceeds $50,000 or not based on demographic and employment-related attributes such as age, education, occupation, working hours, and marital status.

This is a binary classification problem, where the target variable is income level (<=50K or >50K). Accurate income prediction can help in socioeconomic analysis, policy making, and decision support systems.


DATASET DESCRIPTION

The dataset used in this project is the Adult Census Income Dataset, originally sourced from the UCI Machine Learning Repository.

Total instances: 32,561

Total attributes: 15

Target variable: income

<=50K → Low income

>50K → High income

Key Features:

Age

Workclass

Education

Marital Status

Occupation

Relationship

Race

Sex

Capital Gain

Capital Loss

Hours per week

Native country

The dataset contains both numerical and categorical attributes, and also includes missing values represented as ?.


DATA CLEANING STEPS

The following data preprocessing and cleaning steps were performed:

Handling Missing Values

Missing values represented as ' ?' were replaced with NaN.

Rows containing missing values were removed using dropna().

Removing Duplicate Records

Duplicate rows were identified and removed to avoid biased learning.

Target Variable Encoding

The income column was mapped as:

<=50K → Low

>50K → High

Categorical Feature Encoding

All categorical attributes were converted into numerical form using One-Hot Encoding (get_dummies).

drop_first=True was used to avoid the dummy variable trap.

Feature Scaling

Numerical features were standardized using StandardScaler.

This step is essential for distance-based and margin-based algorithms like KNN and SVM.


ALGORITHMS USED

The following machine learning algorithms were implemented to analyze and predict income levels:

1. Linear Regression

Linear Regression was used as a baseline model to understand the relationship between input features and income. Although it is primarily a regression algorithm, it was applied to observe how continuous predictions behave when mapped to a binary income classification problem.

2. Decision Tree Classifier

Decision Tree is a rule-based algorithm that splits data using feature-based conditions. It is easy to interpret and handles both numerical and categorical features efficiently.

3. Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting. It provides better generalization compared to a single decision tree.

4. K-Nearest Neighbors (KNN)

KNN is a distance-based supervised learning algorithm that classifies data points based on the majority class of their nearest neighbors. Feature scaling was applied to ensure fair distance calculations.

5. Support Vector Machine (SVM)

SVM is a margin-based classification algorithm that finds an optimal hyperplane to separate income classes. The RBF kernel was used to handle non-linear decision boundaries, making it effective for complex feature spaces.


EVALUATION METRICS AND RESULTS

To evaluate the performance of all models, the following metrics were used:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

🔹 Linear Regression

Used as a reference baseline model.

Performance was lower compared to classification-based models.

Highlighted the limitations of regression techniques for classification tasks.

🔹 Decision Tree

Provided clear interpretability of decision rules.

Prone to overfitting without pruning.

Performance was competitive but slightly less stable than ensemble models.

🔹 Random Forest

Achieved the best overall performance among all models.

Reduced overfitting through ensemble learning.

Showed improved precision and recall for both income classes.

🔹 K-Nearest Neighbors (KNN)

Achieved reasonable accuracy after feature scaling.

Sensitive to the choice of k and class imbalance.

Performed well for majority class predictions.

🔹 Support Vector Machine (SVM)

Achieved high generalization accuracy (~84%) on test data.

Performed better than KNN due to its ability to handle non-linear data.

Class imbalance affected recall for the high-income class.


CONCLUSION/OBSERVATIONS

The Support Vector Machine model achieved good predictive performance on the Adult Census dataset.

Proper data preprocessing significantly improved model accuracy.

Class imbalance affected recall for the high-income category, which is a common issue in real-world datasets.

SVM performed better than KNN in terms of overall generalization accuracy.

Feature scaling and encoding played a crucial role in improving model performance.

Overall, the project successfully demonstrates the application of machine learning techniques for income prediction using real-world census data.
