"""TSVC tsvc_2 kernel ``vtv`` (numpy reference)."""


def vtv(a, b, LEN_1D):
    # array shapes (numpy->dace): a=(LEN_1D,), b=(LEN_1D,)
    for i in range(LEN_1D):
        a[i] = a[i] * b[i]
