from sklearn.metrics import mean_squared_error
from math import sqrt

def evaluate_arima_model(series, arima_order):
    model = ARIMA(series, order=arima_order)
    model_fit = model.fit()
    mse = mean_squared_error(series, model_fit.fittedvalues)
    return sqrt(mse)

def grid_search(series, p_values, d_values, q_values):
    best_score, best_params = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    rmse = evaluate_arima_model(series, (p, d, q))
                    if rmse < best_score:
                        best_score, best_params = rmse, (p, d, q)
                except:
                    continue
    return best_params

# Define the parameter search space
p_values = range(0, 6)
d_values = range(0, 3)
q_values = range(0, 6)

# Find the optimal parameters
optimal_order = grid_search(data['SpotPriceEUR'][0:500], p_values, d_values, q_values)
