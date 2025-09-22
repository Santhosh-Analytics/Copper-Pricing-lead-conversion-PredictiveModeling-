<div align="center">
    <img alt="Static Badge" src="https://img.shields.io/badge/Modeling-blue?style=for-the-badge&logoColor=white&logo=Python&label=Copper&labelColor=grey&color=blue">
</div>

# <div align="center"> Copper Modeling Project</div>

<div align="center"> A comprehensive machine learning project to analyze and model the copper industry data, addressing challenges such as skewness, noisy data, and lead classification. </div>

## Table of Contents

<a href="#overview"><img alt="Static Badge" src="https://img.shields.io/badge/Overview_-blue?style=--&logo=headspace&logoColor=maroon&ogoSize=auto" style="font-size: 30px; font-weight: bold;"></a> &nbsp;
<a href="#problem-statement"><img alt="Static Badge" src="https://img.shields.io/badge/Problem%20Statement_-blue?style=--&logo=headspace&logoColor=maroon"></a> &nbsp;
<a href="#Outcome"><img alt="Static Badge" src="https://img.shields.io/badge/Problem%20Statement_-blue?style=--&logo=headspace&logoColor=maroon"></a> &nbsp;
<a href="#data-source"><img alt="Static Badge" src="https://img.shields.io/badge/Data%20Source_-blue?style=--&logo=headspace&logoColor=maroon"></a> &nbsp;
<a href="#prerequisites"><img alt="Static Badge" src="https://img.shields.io/badge/Prerequisites_-blue?style=--&logo=headspace&logoColor=maroon"></a> &nbsp;
<a href="#installation"><img alt="Static Badge" src="https://img.shields.io/badge/Installation_-blue?style=--&logo=headspace&logoColor=maroon"></a> &nbsp;
<a href="#usage"><img alt="Static Badge" src="https://img.shields.io/badge/Usage_-blue?style=--&logo=headspace&logoColor=maroon"></a> &nbsp;
<a href="#models"><img alt="Static Badge" src="https://img.shields.io/badge/Models_-blue?style=--&logo=headspace&logoColor=maroon"></a> &nbsp;
<a href="#streamlit-gui"><img alt="Static Badge" src="https://img.shields.io/badge/Streamlit%20GUI_-blue?style=--&logo=headspace&logoColor=maroon"></a> &nbsp;
<a href="#contributing"><img alt="Static Badge" src="https://img.shields.io/badge/Contributing_-blue?style=--&logo=headspace&logoColor=maroon"></a> &nbsp;
<a href="#license"><img alt="Static Badge" src="https://img.shields.io/badge/License_-blue?style=--&logo=headspace&logoColor=maroon"></a>
<a href="#contact"><img alt="Static Badge" src="https://img.shields.io/badge/Contact_-blue?style=--&logo=headspace&logoColor=maroon"></a>

## Overview

This project aims to develop machine learning models to address the challenges faced by the copper industry in pricing and lead classification. By leveraging advanced techniques such as data normalization, feature scaling, and outlier detection, the project delivers robust solutions for accurate pricing decisions and effective lead classification.

## Problem Statement

The project addresses the following objectives:

1. Explore skewness and outliers in the copper industry dataset.
2. Transform the data into a suitable format and perform necessary cleaning and pre-processing steps.
3. Build a machine learning regression model to predict the continuous variable 'Selling_Price'.
4. Develop a machine learning classification model to predict lead status (WON or LOST).
5. Evaluate the model using appropriate evaluation metrics such as accuracy, precision, recall, F1 score, and AUC curve for Classification problem and R2 Score, Error metrics for Regression problem.
6. Regularize the model by leveraging advanced techniques such as cross-validation and grid search to find the best-performing model.
7. Create a Streamlit application where users can input column values and obtain predicted 'Selling_Price' or 'Status' (WON/LOST).

## Outcome:

1.Tech Stack: Python, Streamlit, Docker, XGBoost, Git, GitHub

1.Developed a dual-purpose ML system for industrial sales optimization: regression for copper price prediction and classification for lead conversion.

1.Achieved 0.96/0.99 accuracy and 0.97/0.99 F1-score (classification); 0.94/0.95 R² and RMSE of 24.23/22.43 (regression).

1.Engineered 12 domain-specific features and addressed a 1:4 class imbalance using SMOTETomek, improving minority class recall.

1.Deployed real-time prediction app with Streamlit and Docker, enabling interactive inference for copper pricing and lead conversion scoring.

1.Reduced pricing decision time from 2 hours to 2 minutes, significantly improving operational efficiency.

## Data Source

The copper industry dataset used in this project should be provided or specified.

## Prerequisites

Before running the project, ensure you have the following prerequisites installed:

- Python: Version 3.07 or higher. <a href="https://www.python.org/downloads">
  <img alt="Static Badge" src="https://img.shields.io/badge/Download_Python-darkgreen?style=--&logo=python&logoColor=white">
  </a>

- Required packages : <img alt="Static Badge" src="https://img.shields.io/badge/Pandas--NumP--SKlearn--Seaborn--Matplotlib--Streamlit-darkgreen?style=--&logo=pypi&logoColor=white">

- Install dependencies: <img alt="Static Badge" src="https://img.shields.io/badge/pip install --r requirements.txt-darkgreen?style=--&logo=pypi&logoColor=white">

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Santhosh-Analytics/Copper-Pricing-lead-conversion-PredictiveModeling-.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Copper-Pricing-lead-conversion-PredictiveModeling
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Prepare the data:

Obtain the copper industry dataset and place it in the appropriate directory.

## Run the data preprocessing scripts:

Execute the scripts to handle missing values, treat outliers, and address skewness in the dataset.
Perform feature engineering and encoding as required.

## Train the models:

Run the scripts to train the regression model for predicting 'Selling_Price'.
Train the classification model for predicting lead status (WON or LOST).

## Launch the Streamlit application:

Execute the Streamlit script to start the interactive application.
Input the required column values and obtain predictions for 'Selling_Price' or 'Status' (WON/LOST).

## Models

The project includes the following machine learning models:

Regression model: [Model name(s)] for predicting the continuous variable 'Selling_Price'.
Classification model: [Model name(s)] for predicting lead status (WON or LOST).

## Streamlit GUI

The project includes a user-friendly Streamlit application that allows users to input column values and obtain predictions for 'Selling_Price' or 'Status' (WON/LOST) based on the trained models.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

If you have any questions or need further assistance, feel free to reach out to the project maintainers:

Name: [Your Name]
Email: [your-email@example.com]
