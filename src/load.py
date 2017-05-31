from scipy.sparse import dok_matrix


def load_matrix(f):
    with open(f, "r") as f_mat:
        mat = dok_matrix(tuple(map(int, f_mat.readline().strip().split())))
        for line in f_mat:
            p, t, v = map(float, line.strip().split())
            mat[p, t] = v
    return mat


def load_pair2node(f):
    return map(lambda x: tuple(map(int, x.strip().split())), open(f, "r"))


def load_truth(f):
    return map(lambda x: float(x.strip()), open(f, "r"))
