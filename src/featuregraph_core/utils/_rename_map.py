"""
bidmc_map transcribed from the real state_detection/utils/_rename_map.py
(private repo, commit 8bfee18), lines 72-79.

NOTE ON SCOPE: the real file also contains a second dict, eastman_map,
mapping Tennessee Eastman Process (TEP) column names (xmeas_1..41,
xmv_1..11, faultNumber, simulationRun) to descriptive names. That dict
is deliberately NOT included here -- it's irrelevant to this package's
purpose (reproducing the smoothing paper's BIDMC-only population
statistics), and TEP work hasn't started yet regardless. It was also
only partially visible when this package was assembled (lines 1-19 and
45-79 of the real file were seen; lines 20-44 were not), so it
couldn't be transcribed completely even if it were in scope.
"""

bidmc_map = {
    "Time [s]": "time",
    "RESP": "respiration",
    "PLETH": "ppg",
    "II": "ecg_ii",
    "V": "ecg_v",
    "AVR": "ecg_avr",
}
