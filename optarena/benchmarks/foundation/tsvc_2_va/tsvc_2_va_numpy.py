"""TSVC tsvc_2 kernel ``va`` (numpy reference)."""


def va(a, b, LEN_1D):
    # array shapes (numpy->dace): a=(LEN_1D,), b=(LEN_1D,)
    for i in range(LEN_1D):
        a[i] = b[i]
