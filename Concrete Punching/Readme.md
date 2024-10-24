# Punching Shear Resistance Prediction

![Project Banner](https://img.shields.io/badge/Punching-Shear%20Resistance-blue)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

![Punching](https://github.com/user-attachments/assets/84341d3a-6f88-41bb-97b8-a641d12d0113)

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Data Preparation](#data-preparation)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Machine Learning Model](#machine-learning-model)
- [Project Goals](#project-goals)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Welcome to the **Punching Shear Resistance Prediction** project! This repository contains a Python script that leverages machine learning to predict the punching shear resistance (`Pu`) in concrete slab-column connections. By analyzing various geometric and material properties, the model provides engineers with a reliable tool to assess structural integrity and prevent potential failures.

## Problem Statement

**Concrete Punching Shear** is a failure mode in slab-column connections where concentrated loads cause localized perforations, leading to potential structural collapse. Accurately predicting the punching shear resistance is crucial for designing safe and economical reinforced concrete structures. Traditional methods rely on empirical formulas, which may not capture the complexities of real-world scenarios.

## Solution

This project addresses the problem by developing a regression-based machine learning model to predict the punching shear resistance (`Pu`). The approach involves:

1. **Data Collection & Preparation**: Importing and cleaning the dataset containing measurements of various geometric and material properties related to slab-column connections. The database for the project was obtained from the Kaggle: [Dataset](https://www.kaggle.com/datasets/jrsuri/punching-shear-of-flat-concrete-slabs)
2. **Feature Engineering**: Creating new features such as the Basic Control Perimeter (`bcp`) and Reinforcement Area (`reinforcement_area`) to enhance model performance.
3. **Exploratory Data Analysis (EDA)**: Visualizing relationships between features and the target variable to inform model selection and feature importance.
4. **Model Training & Validation**: Building and evaluating multiple regression models, including Linear Regression, Gradient Boosting Regressor, Bayesian Ridge, and XGBoost Regressor, to determine the best-performing model.
5. **Prediction**: Providing a mechanism to input new data and obtain predictions for punching shear resistance.

## Data Preparation

The dataset used in this project includes the following features:

- **Shape**: Cross-section shape of the column (`S`=Square, `C`=Circular, `R`=Rectangular)
- **b1 (mm)**: Column side or smaller side if Shape is `R`
- **d1 (mm)**: Larger side of the column if Shape is `R`
- **davg (mm)**: Average effective depth in X and Y directions
- **ravg**: Average reinforcement ratio in X and Y directions
- **b\* (mm)**: Column effective width
- **b*/davg**: Effective width divided by effective depth
- **fc (MPa)**: Concrete compressive strength
- **fy (MPa)**: Steel yield strength
- **Pu (kN)**: Punching shear resistance (target variable)

Data preprocessing steps include handling missing values, removing duplicates, and engineering new features to better capture the underlying patterns.

## Exploratory Data Analysis (EDA)

EDA was performed to understand the relationships between different features and the target variable (`Pu`). Key observations include:

- Positive correlation between slab thickness (`davg`) and punching shear resistance.
- The Basic Control Perimeter (`bcp`) and Reinforcement Area (`reinforcement_area`) are strong predictors of `Pu`.
- Heatmaps reveal multicollinearity among certain geometric features.

These insights guided the feature selection and engineering process, ensuring that the model captures the most relevant information.

## Machine Learning Model

Several regression models were evaluated to predict `Pu`, including:

1. **Linear Regression**
2. **Gradient Boosting Regressor**
3. **Bayesian Ridge**
4. **XGBoost Regressor**

The modeling pipeline includes:

- **Feature Selection**: Utilizing Univariate Feature Selection and Recursive Feature Elimination with Cross-Validation (RFECV) to identify the most impactful features.
- **Preprocessing**: Applying One-Hot Encoding for categorical variables and scaling numerical features.
- **Hyperparameter Tuning**: Using GridSearchCV to find the best combination of preprocessing steps and model parameters.
- **Validation**: Assessing model performance using Median Absolute Error (MAE) and predictive accuracy metrics.

The best-performing model offers accurate and reliable predictions, making it a valuable tool for structural engineers.

## Project Goals

- **Accurate Prediction**: Develop a robust model to accurately predict punching shear resistance based on geometric and material properties.
- **Feature Engineering**: Create meaningful features that enhance model performance and interpretability.
- **Model Optimization**: Explore and tune various machine learning algorithms to achieve optimal predictive accuracy.
- **Practical Application**: Provide a user-friendly interface for engineers to input data and receive instant predictions.

## Installation

To run the code in this repository, ensure you have the following prerequisites:

- **Python 3.8 or higher**
- **Google Colab** (for running the notebook, or alternatively set up a local environment)

### Clone the Repository

```bash

git clone https://github.com/grzegorz-gomza/Portfolio_Projects/tree/main/Concrete%20Punching.git
cd 'Concrete Punching'
```

### Install Dependencies

If running locally, create a virtual environment and install the required packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Alternative run the following pip command in terminal:

```bash
pip install matplotlib==3.9.0 \
matplotlib-inline==0.1.7 \
numpy==1.26.4 \
pandas==2.2.2 \
scikit-learn==1.5.0 \
seaborn==0.13.2 \
xgboost==2.0.3
```

## Usage

The primary script for this project is `Punching.ipynb`, which can be run in a Jupyter Notebook environment such as Google Colab.

### Running on Google Colab

1. **Open Google Colab**: Navigate to [Google Colab](https://colab.research.google.com/) and sign in with your Google account.
2. **Mount Google Drive**: The script accesses data from Google Drive. Ensure your dataset is uploaded to your Drive.
3. **Upload the Script**: Upload `Punching.ipynb` to Colab or copy the content into a new Colab notebook.
4. **Execute the Cells**: Run each cell sequentially to perform data preparation, EDA, model training, and prediction.
5. **Make Predictions**: Use the example input provided at the end of the script to estimate `Pu` for new data points.

### Running Locally

1. **Ensure Dependencies are Installed**: As per the installation instructions.
2. **Update Data Paths**: Modify the script to point to the local paths where the dataset is stored.
3. **Run the Script**: Execute the Python script using a Jupyter Notebook or your preferred IDE.

## Results

After training, the model was validated using cross-validation techniques and demonstrated strong predictive performance with low Median Absolute Error (MAE). The final model can accurately estimate `Pu` based on input features, aiding in the design and assessment of concrete structures.

### Example Prediction

The preditions are made in the cell contaning the following code:

```python
# Example input data
sample_input = {
    'Shape': ['C'],             # Circular Column
    'davg (mm)': [121],         # Static slab height
    'fc (MPa)': [23.5],         # Concrete strength
    'fy (MPa)': [450],          # Reinforcement strength
    'bcp (mm)': [2165],         # Basic control perimeter
    'reinforcement_area (mm2)': [808] # Reinforcement Area
}

# Making a prediction using the model
predictions = model.predict(pd.DataFrame(sample_input))
print(f'Estimated Pu[kN]: {np.round(predictions,2)[0]}')
```

The data has to be given in brackets []. After running the code, the following output will be displayed:

**Output:**
```
Estimated Pu[kN]: 1234.56
```

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

1. **Fork the Repository**
2. **Create a Feature Branch**
3. **Commit Your Changes**
4. **Push to the Branch**
5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any inquiries or feedback, please contact:

- **Name**: Grzegorz Gomza 
- **Email**: gomza.datascience@gmail.com
- **LinkedIn**: [My LinkedIn](https://www.linkedin.com/in/gregory-gomza/))
- **GitHub**: [grzegorz-gomza](https://github.com/grzegorz-gomza/))

### Bonus 😉
The name "Grzegorz" can be phonetically spelled for a non-Polish person as "Gzheh-gohz."

But you can call me Gregory or Gregor 😁

---

*Happy Predicting!*





