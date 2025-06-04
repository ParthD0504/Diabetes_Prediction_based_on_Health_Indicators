This project aims to harness a comprehensive
dataset of health indicators to develop an effective predictive model that categorizes individuals as
diabetic, pre-diabetic, or healthy. The dataset, sourced from the CDC Diabetes Health Indicators
collection, incorporates 21 diverse features ranging from demographic details to lifestyle and clinical
measures providing a holistic view of the factors influencing diabetes risk.

Our methodological framework begins with a rigorous exploratory data analysis (EDA), where we
assess data integrity, address missing values, and uncover hidden patterns through statistical
summaries and visualizations. Following data preprocessing and feature engineering, we establish a
baseline for performance using well-understood models, such as logistic regression and support
vector machines (SVM). To further enhance predictive accuracy especially in terms of capturing cases
with a high risk of diabetes we implement an advanced model based on the XGBoost algorithm, which
demonstrates significant improvements in recall.

The dataset employed in this project is derived from the CDC Diabetes Health Indicators collection
available on the UCI repository
(https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators). The project leverages
the imbalanced dataset, which more accurately reflects the true distribution of diabetes
occurrences in the surveyed population

In this project, we built the classification models from the ground up by implementing the underlying
mathematical equations and optimization routines. Our models include:
a) Logistic Regression
b) Support Vector Machines (SVM)
c) XGBoost

