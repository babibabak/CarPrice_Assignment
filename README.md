# CarPrice_Assignment
A car price prediction system built with Python and TensorFlow, utilizing the CarPrice_Assignment dataset to predict car prices based on features like engine size, horsepower, and fuel type. The project includes data preprocessing, feature encoding, and training of deep neural networks with varying architectures to optimize prediction accuracy
# Car Price Prediction System
## Overview
This project implements a car price prediction system using the CarPrice_Assignment dataset. It employs deep neural networks built with TensorFlow to predict car prices based on features such as engine size, horsepower, fuel type, and more. The system preprocesses the dataset, encodes categorical features, and evaluates multiple neural network architectures (varying layers and neurons) to optimize prediction accuracy. Performance is measured using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). The project uses Python with libraries like Pandas, NumPy, Scikit-learn, TensorFlow, and Matplotlib for data processing, modeling, and visualization.

## Features
- Data Preprocessing: Loads and cleans the CarPrice_Assignment dataset, handling categorical variables via LabelEncoder and scaling numerical features with StandardScaler.
- Feature Encoding: Converts categorical features (e.g., fuel type, car body) into numerical representations for model training.
- Neural Network Models: Implements and compares deep and wide-and-deep neural network architectures with varying layers (1, 2, 5, 7) and neuron configurations (e.g., 128, 512).
- Model Evaluation: Assesses model performance using MAE and RMSE on a validation set, with results visualized using Matplotlib.
- Visualization: Plots MAE against the number of layers to analyze model performance trends.

## Dataset
The project uses the CarPrice_Assignment dataset (`CarPrice_Assignment.csv`), which contains 205 entries with 26 features, including:

- Numerical Features: `wheelbase`, `carlength`, `carwidth`, `carheight`, `curbweight`, `enginesize`, `boreratio`, `stroke`, `compressionratio`, `horsepower`, `peakrpm`, `citympg`, `highwaympg`, `price` (target variable).
- Categorical Features: `CarName`, `fueltype`, `aspiration`, `doornumber`, `carbody`, `drivewheel`, `enginelocation`, `enginetype`, `cylindernumber`, `fuelsystem`.
- Key Columns:
  - `car_ID`: Unique identifier for each car.
  = `price`: Target variable (car price in USD).
  = Example features: `enginesize` (engine size in cubic inches), `horsepower` (engine power), `fueltype` (gas or diesel).

## Requirements
- Python 3.11
- Libraries: `pandas`, `numpy`, `tensorflow`, `scikit-learn`, `matplotlib`

## Installation
- Clone the repository:
```bash
git clone https://github.com/yourusername/car-price-prediction.git
```

2.Install dependencies:
```bash
pip install -r requirements.txt
```

3.Run the Jupyter Notebook:
```bash
jupyter notebook CarPrice_Assignment.ipynb
```

## Usage
- Load and preprocess the CarPrice_Assignment.csv dataset.
- Encode categorical features and scale numerical features.
- Train neural network models with different architectures (e.g., deep model with 128 and 64 neurons, wide-and-deep model with 128 and 512 neurons).
- Evaluate models on the validation set using MAE and RMSE.
- Visualize the impact of the number of layers on MAE using Matplotlib.

## Example
The project trains multiple models:
- Deep Model: 2 hidden layers (128, 64 neurons), achieving a validation MAE of approximately 2320.25.
- Wide-and-Deep Model: 2 hidden layers (128, 512 neurons), achieving a validation MAE of approximately 1944.64.
- A plot of MAE vs. number of layers (1, 2, 5, 7) is generated to compare model performance.

## Methodology

- Data Preprocessing: Loads the dataset, drops the car_ID column, encodes categorical features using LabelEncoder, and scales numerical features using StandardScaler.
- Model Architecture:
  - Models are built using TensorFlow's Sequential API with ReLU activation functions.
  - Varying architectures are tested: single-layer, multi-layer (1, 2, 5, 7 layers), and wide-and-deep configurations.
- Training: Models are trained for 100 epochs with a batch size of 32, using the Adam optimizer and mean absolute error loss.
- Evaluation: Validation MAE and RMSE are computed to assess model performance.
- Visualization: A line plot visualizes MAE trends across different layer counts.

## Future Improvements
- Incorporate feature engineering to include interactions between features (e.g., horsepower-to-weight ratio).
- Experiment with hyperparameter tuning (e.g., learning rate, batch size) using grid search.
- Add regularization techniques (e.g., dropout, L2 regularization) to prevent overfitting.
- Explore other algorithms like XGBoost or Random Forest for comparison.
- Enhance visualizations with feature importance analysis or prediction error distributions.

