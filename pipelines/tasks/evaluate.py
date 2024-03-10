from math import sqrt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from dslabs import plot_forecasting_series_on_ax
from pipelines import linear_regression

def evaluate(train=None, test=None, predicted_train=None, predicted_test=None):
    test = {
        "RMSE": round(sqrt(mean_squared_error(test, predicted_test)), 2),
        "MAE": round(mean_absolute_error(test, predicted_test), 2),
        "MAPE": round(mean_absolute_percentage_error(test, predicted_test), 2),
        "R2": round(r2_score(test, predicted_test), 2),
    }

    if predicted_train is not None:
        return {
            "train": {
                "RMSE": round(sqrt(mean_squared_error(train, predicted_train)), 2),
                "MAE": round(mean_absolute_error(train, predicted_train), 2),
                "MAPE": round(
                    mean_absolute_percentage_error(train, predicted_train), 2
                ),
                "R2": round(r2_score(train, predicted_train), 2),
            },
            "test": test,
        }

    return {
        "test": test
    }


def compare_with_linear_reg(df, target, ax=None, plot_subtitle=""):
    train, test, predicted_train, predicted_test = linear_regression.run(df, target, path=None)
    plot_forecasting_series_on_ax(
        train,
        test,
        predicted_test,
        ax=ax,
        title=plot_subtitle,
        xlabel=train.index.name,
        ylabel=""
    )
    return evaluate(train, test, predicted_train, predicted_test)
