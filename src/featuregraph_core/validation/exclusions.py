"""
Subjects flagged in the private repo's earlier analysis as showing a
validation gap that does not close as the matching tolerance widens --
a structural construction-annotation mismatch, not ordinary timing
imprecision. Transcribed verbatim (FLAGGED_ANOMALY_SUBJECTS) from the
private repo's ground-truth validation notebook (cell-bidmc-gap-to-ceiling).

Excluding these from the paper's reported recall/precision figures is
the same treatment already given to the 16 BIDMC subjects with no
reliable ACF period at all -- a recorded, structural fact, not an
error hidden from the reader.
"""

FLAGGED_ANOMALY_SUBJECTS = [9, 10, 23, 48, 51]
