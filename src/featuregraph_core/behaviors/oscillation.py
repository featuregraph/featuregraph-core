from dataclasses import dataclass
from featuregraph_smoothing_core.operators.states import rising_state, falling_state
from featuregraph_smoothing_core.operators.events import enter_state, exit_state, event_id, event_index
from featuregraph_smoothing_core.operators.measures import smooth, smooth_grouped

import pandas as pd
import numpy as np

@dataclass
class OscillationConfig:
    signal: str
    eps: float = 0
    lag: int = 1
    smooth_window: int = 100

    @property
    def smooth(self):
        return f'{self.signal}_smooth'

    @property
    def rising_col(self):
        return f'{self.signal}_rising'

    @property
    def falling_col(self):
        return f'{self.signal}_falling'

    @property
    def enter_rising_col(self):
        return f'enter_{self.signal}_rising'

    @property
    def exit_rising_col(self):
        return f'exit_{self.signal}_rising'

    @property
    def peak_event_id_col(self):
        return f'{self.signal}_peak_event_id'

    @property
    def trough_event_id_col(self):
        return f'{self.signal}_trough_event_id'

    @property
    def peak_index_col(self):
        return f'{self.signal}_peak_index'

    @property
    def trough_index_col(self):
        return f'{self.signal}_trough_index'

    def add_primitives(self, df: pd.DataFrame, group=None):
        df = df.copy()
        df[self.smooth] = smooth_grouped(df, self.signal, window=self.smooth_window, group=group)
        df[self.rising_col] = rising_state(df[self.smooth])
        df[self.falling_col] = falling_state(df[self.smooth])
        df[self.exit_rising_col] = exit_state(df[self.rising_col], group=df[group] if group else None)
        df[self.enter_rising_col] = enter_state(df[self.rising_col], group=df[group] if group else None)
        df[self.peak_event_id_col] = event_id(df, self.exit_rising_col, group=group)
        df[self.trough_event_id_col] = event_id(df, self.enter_rising_col, group=group)
        df[self.peak_index_col] = event_index(df, self.exit_rising_col, group=group)
        df[self.trough_index_col] = event_index(df, self.enter_rising_col, group=group)
        return df

    def summarize(self, df: pd.DataFrame, group=None):
        summarydf = df.groupby(group).agg(
            start_index=(self.trough_index_col, 'first'),
            rising_duration=(self.rising_col, 'sum'),
            falling_duration=(self.falling_col, 'sum'),
            peak_index=(self.peak_index_col, 'last'),
            trough_index=(self.trough_index_col, 'first'),
            max_raw_signal=(self.signal, 'max'),
            max_smooth_signal=(self.smooth, 'max'),
            min_raw_signal=(self.signal, 'min'),
            min_smooth_signal=(self.smooth, 'min'),
        )

        summarydf['end_index'] = summarydf.groupby(level=0)['start_index'].shift(-1)
        summarydf['is_complete'] = summarydf['start_index'].notna() & summarydf['end_index'].notna()
        summarydf['duration'] = summarydf['rising_duration'] + summarydf['falling_duration']

        if not group:
            summarydf['period'] = summarydf['peak_index'].diff()
        else:
            summarydf['period'] = summarydf.groupby(group[0])['peak_index'].diff()

        net_change_raw = (summarydf['max_raw_signal'] - summarydf['min_raw_signal'])
        net_change_smooth = (summarydf['max_smooth_signal'] - summarydf['min_smooth_signal'])

        summarydf['amplitude_raw'] =  net_change_raw / 2
        summarydf['amplitude_smooth'] = net_change_smooth / 2

        summarydf['raw_rising_mean_rate'] = (net_change_raw / summarydf['rising_duration']).where(summarydf['rising_duration'] > 0)
        summarydf['raw_falling_mean_rate'] = (net_change_raw / summarydf['falling_duration']).where(summarydf['falling_duration'] > 0)

        summarydf['smooth_rising_mean_rate'] = (net_change_smooth / summarydf['rising_duration']).where(summarydf['rising_duration'] > 0)
        summarydf['smooth_falling_mean_rate'] = (net_change_smooth / summarydf['falling_duration']).where(summarydf['falling_duration'] > 0)

        summarydf["temporal_symmetry"] = (1 - (summarydf['rising_duration'] - summarydf['falling_duration']).abs() / summarydf['duration']).where(summarydf['duration'] > 0)

        return summarydf
