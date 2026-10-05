def smooth(signal, window):
    return (
        signal.rolling(center=True, window=window).median().rolling(center=True, window=window).mean()
    )

def smooth_grouped(df, signal_col, window, group):
    return df.groupby(group)[signal_col].transform(lambda s: smooth(s, window))

def group_transform(df, signal, op, group):
    return df.groupby(group)[signal].transform(op)

def group_map(df, signal, op, group, offset=0):
    return df[group].map(df.groupby(group)[signal].agg(op).shift(offset))
