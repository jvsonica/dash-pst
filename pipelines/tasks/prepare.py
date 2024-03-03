from typing import TypedDict, Optional
from utils import get_options_with_default
from preprocess import aggregation_func_by_col
from dslabs import series_train_test_split
from transformation import smoothing


class PrepareOptions(TypedDict):
    training_pct: Optional[float]
    smoothing: Optional[bool]


DEFAULT_PREPARE_OPTIONS : PrepareOptions = {
    'training_pct': 0.8,
    'smoothing': False,
}


def prepare(df, options: PrepareOptions = DEFAULT_PREPARE_OPTIONS.copy()):
    """Prepare `df` dataset as described in `options`. Preparation includes applying
    smoothing and separating in train and test sets.


    Args:
        df (DataFrame): _description_
        options (dict | None, optional): _description_. Defaults to None.

    Returns:
        tuple: train and test sets.
    """
    options = get_options_with_default(options, default=DEFAULT_PREPARE_OPTIONS)

    # Test/Train Split
    train, test = series_train_test_split(data=df, trn_pct=options['training_pct'])

    # Smooth on training set
    if 'smoothing' in options and options['smoothing']:
        # agg_func = aggregation_func_by_col[target]
        train = smoothing.run(train, window=options['smoothing']['window'], agg_func=aggregation_func_by_col)
        train = train[~train.isna()].copy()

    return train, test
