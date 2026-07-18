"""TSVC tsvc_2 kernel ``s431`` (numpy reference)."""


def s431(a, b, LEN_1D):
    # array shapes (numpy->dace): a=(LEN_1D,), b=(LEN_1D,)
    for i in range(LEN_1D):
        a[i] = a[i] + b[i]
