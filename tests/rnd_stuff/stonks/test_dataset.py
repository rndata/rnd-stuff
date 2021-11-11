from datetime import date, timedelta

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames
from rnd_stuff.stonks.dataset import (
    Dataset,
    DatasetError,
    Sample,
    SimpleDataset
)


good_floats = st.floats(allow_nan=False, allow_infinity=False)


@st.composite
def daily_ix(
        draw,
        start_date=st.dates(
            min_value=date(1990, 1, 1),
            max_value=date(2030, 1, 1),
        ),
        periods=st.integers(min_value=1, max_value=5000)

):
    return pd.date_range(
        start=draw(start_date),
        periods=draw(periods),
        freq='D',
    )


@st.composite
def dataset(
        draw,
        df=data_frames(
            index=daily_ix(),
            columns=[
                column(name='A', elements=good_floats),
                column(name='B', elements=good_floats),
            ]
        ),
        period=st.floats(min_value=0, max_value=1),
):
    df = draw(df)
    period = draw(period)

    days = max(int(len(df) * period), 1)
    delta = pd.Timedelta(days, 'days')

    return Dataset(df, delta)


@given(
    df=data_frames(
        index=daily_ix(),
        columns=[
            column(
                name='A',
                elements=good_floats
            ),
            column(
                name='B',
                elements=good_floats
            ),
        ]
    )
)
@settings(
    max_examples=50,
    deadline=timedelta(milliseconds=200),
)
def test_simple_dataset(df):
    simple_ds = SimpleDataset(df)

    for row in df.itertuples():
        sample = simple_ds.step()

        assert sample.price == row

    with pytest.raises(DatasetError):
        simple_ds.step()


@given(ds=dataset())
@settings(
    max_examples=50,
    deadline=timedelta(milliseconds=1500),
)
def test_dataset(ds):

    days = ds.period.days

    for i in range(2):
        step = 0
        prev_ix = None
        while not ds.closed:
            sample = ds.step()
            step += 1

            prices = list(sample.price[1:])
            ix = sample.price.Index

            if prev_ix:
                assert (ix - prev_ix).days == 1
            prev_ix = ix

            df_prices = ds.prices.loc[ix].to_list()

            assert df_prices == prices

        assert step == days

        with pytest.raises(DatasetError):
            ds.step()

        ds.reset()
