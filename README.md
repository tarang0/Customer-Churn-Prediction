# Customer Churn Prediction

This project analyzes customer data from a telecom company to predict whether a customer is likely to churn or not. It involves end-to-end implementation including data exploration, preprocessing, model training, evaluation, and comparison.

## Dataset

- Dataset used: Telco Customer Churn Dataset ([from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn))
- Target variable: `Churn` (Yes/No)

## Exploratory Data Analysis (EDA)

Used **Matplotlib** and **Seaborn** to analyze feature distributions and relationships:
- Count plots and box plots to visualize customer behavior across contract types, payment methods, and churn
- Identified key trends such as higher churn in monthly contracts and certain internet service types

## Preprocessing

Techniques and libraries used:
- **Missing Value Imputation**: Used `SimpleImputer` with mean strategy for numerical columns
- **Encoding**:
  - One-Hot Encoding (`pd.get_dummies`) for nominal features (e.g., `InternetService`, `Contract`)
  - Label Encoding for binary categorical variables
- **Scaling**: Standardized numerical columns using `StandardScaler`
- **Train/Test Split**: Performed using `train_test_split` with `stratify=y` to maintain class balance
- **Class Imbalance Handling**:
  - Used `class_weight='balanced'` during model training
  - Ensured no data leakage by fitting transformations only on `X_train`

## Models Trained

Implemented and compared the following models using **Scikit-learn**:

| Model              | Accuracy | AUC-ROC |
|-------------------|----------|---------|
| Logistic Regression | ~76%     | ✓       |
| Decision Tree       | ~75%     | ✓       |
| SVM (Linear Kernel) | ~76%     | ✓       |
| **Random Forest**   | **78%**  | ✓       |

- Final model selected: **Random Forest Classifier**

## Evaluation

- Used metrics: **Accuracy**, **Confusion Matrix**, **Classification Report**, and **AUC-ROC Curve**
- Plotted confusion matrix with Seaborn heatmap
- Evaluated precision, recall, F1-score for both classes (Churn / No Churn)

## Technologies Used

- **Languages**: Python
- **Libraries**: 
  - `pandas`, `numpy` (data manipulation)
  - `seaborn`, `matplotlib` (EDA & visualization)
  - `scikit-learn` (modeling, preprocessing, evaluation)

## Results

- Best model (Random Forest) achieved **78% accuracy**
- Balanced performance across classes as shown by the classification report
- AUC-ROC score confirmed good discriminatory ability between churners and non-churners


