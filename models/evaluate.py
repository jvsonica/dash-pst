from math import sqrt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

def evaluate(train, test, predicted_train, predicted_test):
    return {
        'train': {
            'RMSE': round(sqrt(mean_squared_error(train, predicted_train)), 2),
            'MAE': round(mean_absolute_error(train, predicted_train), 2),
            'MAPE': round(mean_absolute_percentage_error(train, predicted_train), 2),
            'R2': round(r2_score(train, predicted_train), 2)
        },
        'test': {
            'RMSE': round(sqrt(mean_squared_error(test, predicted_test)), 2),
            'MAE': round(mean_absolute_error(test, predicted_test), 2),
            'MAPE': round(mean_absolute_percentage_error(test, predicted_test), 2),
            'R2': round(r2_score(test, predicted_test), 2)
        }
    }
