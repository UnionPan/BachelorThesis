import numpy as np


def sample(matrix_l):
    (eig_val, eig_vec) = np.linalg.eig(matrix_l)
    eig_val = np.real(eig_val)
    eig_vec = np.real(eig_vec)
    v_idx = np.where(np.random.rand(eig_val.size) <= (eig_val / (1.0 + eig_val)))
    v = eig_vec[:, v_idx[0]]
    # iterate
    y = np.zeros([len(v_idx[0])])
    for i in range(len(v_idx[0]) - 1, -1, -1):
        p = np.sum(v**2, 1)
        p /= np.sum(p)
        # choose a new item to include
        y[i] = np.where(np.random.random() <= np.cumsum(p))[0][0]
        y[i] = y[i].astype(np.int32)
        j = np.where(v[y[i], :] != 0)[0][0]
        j = j.astype(np.int32)
        v_j = v[:, j]
        v = v[:, list(set(range(v[0, :].size)) - {j})]
        v = v - v_j.reshape([v_j.size, 1]).dot((v[y[i], :] / v_j[y[i]]).reshape([1, v[y[i], :].size]))
        for a in range(i - 1):
            for b in range(a - 1):
                v[:, a] = v[:, a] - v[:, a].transpose().dot(v[:, b]) * v[:, b]
            v[:, a] = v[:, a] / np.linalg.norm(v[:, a])
    y = np.sort(y)
    print y
    return y.astype(np.int32).tolist(), eig_val.tolist()


if __name__ == "__main__":
    tmp_mtx = [[4.51782023619640, -0.0489406660500515, 1.08931041595447, -0.855688513861336, -0.196511696731779],
               [-0.0489406660500515, 0.804187698332478, 0, 0, 0.547836349270015],
               [1.08931041595447, 0, 0.461555764882547, 0, 0],
               [-0.855688513861336, 0, 0, 1.11069259592712, 0],
               [-0.196511696731779, 0.547836349270015, 0, 0, 0.580991974227608]]
    sample(tmp_mtx)

