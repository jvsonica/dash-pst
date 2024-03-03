from typing import TypedDict, Optional
from transformation import scaling, aggregation, differentiation
from utils import get_options_with_default
from preprocess import aggregation_func_by_col


class TransformationOptions(TypedDict):
    scaling: Optional[bool]
    aggregation: Optional[dict]
    differentiation: Optional[bool]


DEFAULT_PREPARE_OPTIONS : TransformationOptions = {
    'scaling': True,
    'aggregation': { 'rule': 'H' },
    'differentiation': False,
}


def transform(df, options: TransformationOptions = DEFAULT_PREPARE_OPTIONS.copy()):
    """Apply transformations to `df` as described in `options`.


    Args:
        df (_type_): _description_
        options (dict | None, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    options = get_options_with_default(options, default=DEFAULT_PREPARE_OPTIONS)

    # Scaling
    if 'scaling' in options and options['scaling']:
        df = scaling.run(df)

    # Aggregation
    if 'aggregation' in options and options['aggregation']:
        agg_func = { k: v for (k,v) in aggregation_func_by_col.items() if k in df.columns }
        df = aggregation.run(df, gran_level=options['aggregation']['rule'], agg_funcs=agg_func)

    # Differentiation
    if 'differentiation' in options and options['differentiation']:
        df = differentiation.run(df)

    return df
