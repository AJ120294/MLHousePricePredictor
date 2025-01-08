# MLHousePricePredictor
A machine learning project focused on predicting house prices using the Ames Housing Dataset. Implements robust preprocessing, feature engineering, and advanced models like LightGBM and XGBoost to achieve a competitive RMSE.

---

# **House Price Prediction Project**

### **Overview**
This project aims to predict house prices using the **Ames Housing Dataset**, focusing on implementing robust data preprocessing, feature engineering, and advanced machine learning models like **LightGBM** and **XGBoost**. The model is evaluated using the **Root Mean Squared Error (RMSE)** metric on the logarithm of `SalePrice`.

---

### **Table of Contents**
1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Workflow](#workflow)
4. [Key Features and Insights](#key-features-and-insights)
5. [Models and Performance](#models-and-performance)
6. [Conclusion](#conclusion)
7. [How to Run the Code](#how-to-run-the-code)
8. [Acknowledgments](#acknowledgments)

---

### **Problem Statement**
The goal of this project is to predict the sale price of houses in the test dataset with the lowest possible RMSE. The metric is calculated as:

\[
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \log(\hat{y}_i) - \log(y_i) \right)^2}
\]

---

### **Dataset**
The dataset includes 81 features describing various attributes of houses in Ames, Iowa. The key features include:
- `OverallQual`: Overall quality of the house.
- `TotalArea`: Combined area of the house (living space, basement, and garage).
- `Neighborhood`: Physical location within Ames city limits.
- `TotalBath`: Total number of bathrooms.

The dataset is split into training and test datasets:
- **Train.csv**: Includes house features and `SalePrice` (target variable).
- **Test.csv**: Includes house features but excludes `SalePrice`.

---

### **Workflow**
1. **Data Preprocessing**:
   - Handled missing values based on feature types (e.g., median for numerical, "None" for categorical).
   - Dropped features with excessive missing data (>50%).
   - Encoded categorical features using **Label Encoding**.

2. **Feature Engineering**:
   - Created new features like `TotalArea` (sum of living area, garage, and basement).
   - Engineered interaction terms such as `HouseQualIdx` (`OverallQual * OverallCond`).

3. **Outlier Handling**:
   - Identified and removed outliers using the **IQR method** for numerical features.

4. **Feature Scaling**:
   - Applied **MinMax Scaling** to normalize numerical features.

5. **Model Training**:
   - Baseline model: **Random Forest**.
   - Advanced models: **LightGBM** and **XGBoost**.
   - Hyperparameter tuning using **GridSearchCV** for optimal performance.

6. **Evaluation**:
   - Validation set used to assess model performance.
   - Final predictions submitted for testing.

---

### **Key Features and Insights**
- **Top Predictors**:
  - `TotalArea`: The single most important feature, with a strong correlation to `SalePrice`.
  - `OverallQual`: Reflects the overall quality of the house and significantly impacts pricing.
  - `Neighborhood`: Captures location-based price variations.
- **Insights**:
  - Larger homes with higher build quality located in desirable neighborhoods tend to have higher prices.

---

### **Models and Performance**
| Model             | RMSE (Validation Set) |
|--------------------|------------------------|
| Baseline Model     | 0.1186                |
| XGBoost            | 0.1199                |
| LightGBM (Tuned)   | **0.1142**            |

---

### **Conclusion**
- The **LightGBM model** performed best, achieving an RMSE of **0.1142** on the validation set.
- Further improvements can focus on:
  - Ensembling multiple models.
  - Exploring additional location-specific features.

---

### **How to Run the Code**
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Train the model and generate predictions:
   - Ensure `train.csv` and `test.csv` are placed in the project folder.
   - Execute all cells to preprocess data, train models, and create a `submission.csv` file.

---

### **Acknowledgments**
- Thanks to Kaggle for providing the **Ames Housing Dataset**.
- Dataset and competition: House Prices - Advanced Regression Techniques
Author(s): Anna Montoya, DataCanary
Source: https://kaggle.com/competitions/house-prices-advanced-regression-techniques
Year: 2016
- Libraries used:
  - **Pandas** and **NumPy** for data manipulation.
  - **Matplotlib** and **Seaborn** for visualizations.
  - **LightGBM**, **XGBoost**, and **Scikit-learn** for modeling.
