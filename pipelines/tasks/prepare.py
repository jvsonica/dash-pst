import pandas as pd
from typing import TypedDict, Optional
from utils import get_options_with_default
from preprocess import aggregation_func_by_col
from dslabs import series_train_test_split, dataframe_temporal_train_test_split
from transformation import smoothing


class PrepareOptions(TypedDict):
    training_pct: Optional[float]
    smoothing: Optional[bool]


DEFAULT_PREPARE_OPTIONS : PrepareOptions = {
    'training_pct': 0.8,
    'smoothing': False,
}


def prepare(data, options: PrepareOptions = DEFAULT_PREPARE_OPTIONS.copy()):
    """Prepare `df` dataset as described in `options`. Preparation includes applying
    smoothing and separating in train and test sets.

    Args:
        data (Series|DataFrame): _description_
        options (dict | None, optional): _description_. Defaults to None.

    Returns:
        tuple: train and test sets.
    """
    options = get_options_with_default(options, default=DEFAULT_PREPARE_OPTIONS)

    # Test/Train Split
    train = None
    test = None

    if type(data) == pd.Series:
        train, test = series_train_test_split(data=data, trn_pct=options['training_pct'])
    elif type(data) == pd.DataFrame:
        train, test = dataframe_temporal_train_test_split(data=data, trn_pct=options['training_pct'])
    else:
        raise(f'Unsupported data type {type(data)}')

    # Smooth on training set
    if 'smoothing' in options and options['smoothing']:
        agg_func = { k: v for (k, v) in aggregation_func_by_col.items() if k in data.columns }
        train = smoothing.run(train, window=options['smoothing']['window'], agg_func=agg_func)
        train = train.dropna()

        # Ensure we return a Series when parameter `data` was a Series.
        if type(data) == pd.Series:
            train = pd.Series(train.iloc[:, 0])

    return train, test
