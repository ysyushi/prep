import numpy as np


# x: 2-D array; obj and jac take 2-D array
def gradient_descent(x_init, obj, jac, eta, infinitesimal, step_limit, step_len_init=0.001, step_thres=1000):
    x = x_init
    obj_x = obj(x)
    step_num = 0
    step_len_const = step_len_init
    diff_ratio = 1.

    while abs(diff_ratio) > eta and step_num < step_limit:
        step_num += 1
        step_len = 1. * step_len_const * step_thres / step_num if step_num >= step_thres else step_len_const

        x_new = simplex_project((x - step_len * jac(x)), infinitesimal)
        obj_x_new = obj(x_new)
        diff_ratio = (obj_x_new - obj_x) / abs(obj_x)

        if diff_ratio > eta:
            step_len_const /= 5.
        else:
            x = x_new
            obj_x = obj_x_new

    # if step_num >= step_limit:
    #     print "Fail to converge! diff_ratio:", diff_ratio, "step_len:", step_len
    return x, obj_x


# y: 2-D array
def simplex_project(y, infinitesimal):
    # 1-D vector version
    # D = len(y)
    # u = np.sort(y)[::-1]
    # x_tmp = (1. - np.cumsum(u)) / np.arange(1, D+1)
    # lmd = x_tmp[np.sum(u + x_tmp > 0) - 1]
    # return np.maximum(y + lmd, 0)

    n, d = y.shape
    x = np.fliplr(np.sort(y, axis=1))
    x_tmp = np.dot((np.cumsum(x, axis=1) + (d * infinitesimal - 1.)), np.diagflat(1. / np.arange(1, d + 1)))
    lmd = x_tmp[np.arange(n), np.sum(x > x_tmp, axis=1) - 1]
    return np.maximum(y - lmd[:, np.newaxis], 0) + infinitesimal
