# 🧠 Employee Attrition Prediction

This project predicts employee attrition using machine learning techniques. The pipeline is developed in **Google Colab**, and the best model is deployed via a **Streamlit** web app, supported by local development using **Spyder IDE**.

---

## 🚀 Project Workflow

### 🧪 Phase 1: Development (Google Colab)

#### 1. Install & Load
- Install required libraries
- Load the dataset into a pandas DataFrame

#### 2. Data Cleaning
- Drop constant and irrelevant columns (e.g., columns with only one unique value)
- Ensure data types and values are consistent
- Handle null values and remove duplicate rows

#### 3. Statistical Tests
- **Chi-Square Test** for categorical features vs target (`Attrition`)
- **T-test** for numerical features vs target

#### 4. Exploratory Data Analysis (EDA)
- Storytelling to understand data context
- Categorical Features → `countplot`
- Numerical Features → separate into discrete/continuous and plot using `countplot` and `boxplot`
- Distribution of numeric data using `histplot`

#### 5. Feature Engineering
- Detect and correct outliers using the **IQR method** (Winsorization)
- Encode categorical features (Label Encoding / One-Hot Encoding)
- Analyze feature correlation with heatmap (handle multicollinearity)

#### 6. Train-Test Split
- Split the data into training and test sets

#### 7. Data Balancing
- Handle class imbalance using **SMOTE**

#### 8. Data Scaling
- **Standardize** continuous features for linear models
- **Skip standardization** for tree-based models

#### 9. Model Training
- Train linear models on scaled data
- Train tree models on unscaled data
- Perform cross-validation on top-performing models
- Fine-tune hyperparameters of the best model

#### 10. Bias-Variance Analysis
- Compare training vs test accuracy
- Plot learning curves

#### 11. Feature Importance
- Use model's built-in importance method
- Apply **SHAP** for deeper explainability

#### 12. Model Saving
- Save the final model using `pickle` or `joblib`

---

### 🌐 Phase 2: Deployment (Streamlit + Spyder IDE)

#### 1. Load the Saved Model
- Load the trained and saved model into Streamlit

#### 2. Input Data via UI
- Collect user input via form and convert to DataFrame

#### 3. Preprocess Input
- Apply same preprocessing as training (scaling, encoding, etc.)

#### 4. Match Feature Columns
- Align input columns with model expectations
- Fill missing encoded columns with `0`

#### 5. Predict
- Predict employee attrition with the trained model

#### 6. Display Probability
- Show prediction probability

#### 7. Provide Suggestions
- Display personalized advice based on prediction

---

## 💡 Key Takeaways

1. **Feature Significance**:
   - Use Chi-square and T-test to prioritize features.
   - Avoid dropping features purely based on p-values; assess them again during modeling.

2. **Visualization Guidelines**:
   - Use `countplot` for discrete features, `boxplot` for continuous features.

3. **Avoid Data Leakage**:
   - Always split dataset into `X` (features) and `y` (target) before feature engineering.

4. **Multicollinearity**:
   - Use correlation heatmaps to identify and handle highly correlated features.

5. **Prediction Input Matching**:
   - Trained models expect processed and encoded features.
   - Ensure future input data is preprocessed similarly and matches model structure.

---
