"""TSVC tsvc_2 kernel ``s000`` (numpy reference)."""


def s000(a, b, LEN_1D):
    # array shapes (numpy->dace): a=(LEN_1D,), b=(LEN_1D,)
    for i in range(LEN_1D):
        a[i] = b[i] + 1.0
