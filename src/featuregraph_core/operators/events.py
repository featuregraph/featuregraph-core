import pandas as pd
import numpy as np

def enter_state(state, group=None):
    x = state.astype(int)
    if group is None:
        return x.diff().eq(1)
    return x.groupby(group).diff().eq(1)


def exit_state(state, group=None):
    x = state.astype(int)
    if group is None:
        return x.diff().eq(-1)
    return x.groupby(group).diff().eq(-1)


def event_id(df, enter_col, group=None):
    if group is None:
        return df[enter_col].cumsum()
    return df.groupby(group)[enter_col].cumsum()

def event_index(df, event_col, group=None):
    event_positions = pd.Series(np.where(df[event_col], df.index, np.nan), index=df.index)
    if group is None:
        return event_positions.ffill()
    return event_positions.groupby(df[group]).ffill()
