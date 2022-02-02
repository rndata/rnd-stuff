"""Portfolio math model
"""
import numpy as np

from sklearn.linear_model import LinearRegression


def poor_mans_realloc(fee, curr_w, buysell, tol=1e-10):
    assert abs(sum(curr_w) - 1) < tol
    assert len(curr_w) == len(buysell)
    assert abs(sum(buysell[buysell > 0]) - 1) < tol

    sell_flags = buysell < 0
    buy_flags = ~sell_flags

    new_w = np.zeros_like(curr_w)

    new_w[sell_flags] = curr_w[sell_flags] * (1 + buysell[sell_flags])

    free_amount = sum(-curr_w[sell_flags] * buysell[sell_flags] * (1 - fee))

    new_w[buy_flags] = \
        curr_w[buy_flags] + buysell[buy_flags] * free_amount * (1 - fee)
    return new_w


def rand_args(k=4):
    fee = np.random.rand(1)[0]
    r = np.random.rand(k)
    weights = r / sum(r)

    bs = np.random.rand(k)
    to_sell_ix = set(np.random.randint(1, k-1, size=k-1))
    to_buy_ix = set(range(k)) - to_sell_ix

    s_ix = list(to_sell_ix)
    b_ix = list(to_buy_ix)

    bs[s_ix] *= -1
    bs[b_ix] = bs[b_ix] / sum(bs[b_ix])
    return fee, weights, bs


def gen_ds(samples=10000, size=4, fee=.01):
    X = []
    y = []
    for i in range(samples):
        _, w, bs = rand_args(size)
        new_alloc = poor_mans_realloc(fee, w, bs)
        new_w = new_alloc / sum(new_alloc)
        k = sum(new_alloc)

        X.append([*w, *new_w])
        y.append(k)

    X = np.array(X)
    y = np.array(y)
    return X, y


def approx_rebalance_coef(
        fee,
        size,
        train_samples=10000,
        test_samples=1000,
):
    X, y = gen_ds(samples=train_samples, size=size, fee=fee)
    X_test, y_test = gen_ds(samples=test_samples, size=size, fee=fee)

    reg = LinearRegression().fit(X, y)

    def fn(w1, w2):
        args = np.stack([w1, w2], axis=1)
        n, i, j = args.shape
        return reg.predict(args.reshape(n, i*j))

    return fn, reg.score(X_test, y_test), reg
