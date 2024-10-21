# Autoregression with Linear Regression

This Python script demonstrates the use of autoregression, where predictions made by a trained linear regression model are fed back as inputs to generate future predictions. The model is trained on a time series dataset generated using a noisy sine wave, and the predictions are visualized along with the original data.

## Features
- **Autoregression**: Uses previous predictions as inputs to forecast future values.
- **Linear Regression**: A closed-form solution of linear regression using the normal equation.
- **Sine Wave Generation**: Simulates a noisy sine wave as the input time series.
- **Mean Squared Error (MSE)**: Computes the MSE between the predicted and actual values for a set number of predictions.
- **Visualization**: Plots both the original sine wave and the model’s predictions.

## Requirements
- `torch`: For tensor operations and linear algebra.
- `matplotlib`: For plotting the sine wave and predictions.
- `numpy`: For numerical operations.
- `argparse`: For command-line argument parsing.

## Usage
Run the script with optional arguments for `tau` (time delay), `num_iter` (number of predictions), and `noise` (amount of noise in the sine wave).

### Example:
```bash
python autoregression_linear_regression.py --tau 10 --num_iter 150 --noise 0.1
```

### Command-line Arguments:
- `tau`: The number of time steps used as input features for prediction (default: 10).
- `num_iter`: The number of future time steps to predict (default: 150).
- `noise`: The level of noise added to the sine wave (default: 0.1).

## Output
- **Plot**: The script generates a plot showing the original sine wave and the predicted values. The plot is saved as a PNG file in the specified directory.
- **Mean Squared Error (MSE)**: The MSE between the actual and predicted values is displayed in the terminal.

## License
This project is open source under the MIT License.
