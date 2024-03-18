import os
from matplotlib.pyplot import savefig
from dslabs import (
    plot_forecasting_series,
    plot_forecasting_eval,
)
from .evaluate import evaluate

def save_report(model, target, train, test, predicted_train, predicted_test, observations=[], path='temp/', title=""):
    # Create target dir if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Get evaluation metrics
    results = evaluate(train, test, predicted_train, predicted_test)
    for metric, value in results['test'].items():
        print(f'{metric}:\t{value}')
    
    # Plot forecast
    plot_forecasting_eval(
        train, test, predicted_train, predicted_test, title=f"{title} {target} ({model})"
    )
    savefig(f"{path}/{model}-{target}-eval.png")

    plot_forecasting_series(
        train,
        test,
        predicted_test,
        title=f"{target} - {model}",
        xlabel=train.index.name,
        ylabel=target,
    )
    savefig(f"{path}/{model}-{target}-forecast.png")

    # Save metrics
    with open(f'{path}/{model}-{target}-run.txt', 'w') as f:
        f.write(f'target: {target}\n')
        f.write(f'start: {train.index[0]}\n')
        f.write(f'end: {train.index[-1]}\n')
        f.write('\n\n')
        f.write('# Metrics Train\n')
        f.write('\n'.join([f'{metric}: {value}' for (metric, value) in results['train'].items()]))
        f.write('\n\n')
        f.write('# Metrics Test\n')
        f.write('\n'.join([f'{metric}: {value}' for (metric, value) in results['test'].items()]))
        if len(observations) > 0:
            f.write('\n\n')
            f.write('# Observations\n')
            f.write('\n'.join([obs for obs in observations]))
