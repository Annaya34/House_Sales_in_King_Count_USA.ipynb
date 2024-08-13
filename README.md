
# House Sales in King County, USA

This project analyzes house sales in King County, USA, using machine learning techniques. The dataset includes house features and sale prices, and the goal is to predict the house prices based on various features.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Modeling](#modeling)
  - [Data Preprocessing](#data-preprocessing)
  - [Linear Regression](#linear-regression)
  - [Ridge Regression with Polynomial Features](#ridge-regression-with-polynomial-features)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [License](#license)

## Overview

This project utilizes multiple linear regression and ridge regression techniques to predict the sale prices of houses in King County, USA. The analysis includes feature engineering, polynomial transformation, and regularization to enhance the model's predictive capabilities.

## Dataset

The dataset is sourced from house sales data in King County, USA. It contains 21,613 observations and 19 features, including:

- `price`: The sale price of the house.
- `bedrooms`: Number of bedrooms.
- `bathrooms`: Number of bathrooms.
- `sqft_living`: Square footage of the living area.
- `sqft_lot`: Square footage of the lot.
- `floors`: Number of floors.
- `waterfront`: A binary indicator if the house is a waterfront property.
- `view`: An index from 0 to 4 of how good the view of the property was.
- `condition`: An index from 1 to 5 on the condition of the house.
- `grade`: An index from 1 to 13, where 1-3 falls short of building construction and design, 7 indicates an average level of construction and design, and 11-13 indicates a high-quality grade.

## Dependencies

To run this project, you will need the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter`

You can install these packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Project Structure

The project files are organized as follows:

- `House_Sales_in_King_Count_USA.ipynb`: Jupyter Notebook containing the analysis and model implementation.
- `README.md`: This file, providing an overview of the project.
- `data/`: Directory containing the dataset (if needed).
- `images/`: Directory to save plots and visualizations (optional).

## Modeling

### Data Preprocessing

The data preprocessing steps include:

- Dropping unnecessary columns (`id`, `Unnamed: 0`).
- Handling missing values (if any).
- Converting categorical variables into numerical representations.
- Splitting the data into training and testing sets.

### Linear Regression

A linear regression model is fitted to predict house prices based on features like `sqft_living`, `grade`, `bathrooms`, `bedrooms`, etc. The model is evaluated using the R² score.

### Ridge Regression with Polynomial Features

To improve model performance, a second-order polynomial transformation is applied to the features. A Ridge regression model is then fitted on this transformed data with a regularization parameter (`alpha`) set to 0.1.

```python
# Polynomial transformation
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Ridge regression with regularization
RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(X_train_poly, y_train)
yhat = RidgeModel.predict(X_test_poly)

# R² score evaluation
print(f"R^2 Score: {r2_score(y_test, yhat)}")
```

## Evaluation

The models are evaluated based on the R² score, which indicates how well the features explain the variance in the target variable (`price`). Higher R² scores imply better predictive performance.

## Usage

To use this project:

1. Clone the repository or download the project files.
2. Ensure that the necessary dependencies are installed.
3. Open `House_Sales_in_King_Count_USA.ipynb` in Jupyter Notebook.
4. Run the cells sequentially to reproduce the analysis and results.

## License

This project is open-source and available under the [MIT License](LICENSE).
