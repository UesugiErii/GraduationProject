import numpy as np
import numba
import time
gamma = 0.99

class A():
    def __init__(self):
        pass

    @staticmethod
    @numba.jit(nopython=True)
    def calc_realv_and_adv_GAE(v, r, done):
        length = r.shape[0]
        num = r.shape[1]

        adv = np.zeros((length + 1, num), dtype=np.float32)

        for t in range(length - 1, -1, -1):
            delta = r[t, :] + v[t + 1, :] * gamma * (1 - done[t, :]) - v[t, :]
            adv[t, :] = delta + gamma * 0.95 * adv[t + 1, :] * (1 - done[t, :])

        adv = adv[:-1, :]

        realv = adv + v[:-1, :]

        return realv, adv


v = np.random.random((65,16)).astype(np.float32)
r = np.random.random((64,16)).astype(np.float32)
done = np.zeros((64,16)).astype(np.float32)

# print(a.calc_realv_and_adv(v, r, done))
a = A()
print(time.time())
a.calc_realv_and_adv_GAE(v, r, done)
print(time.time())
a.calc_realv_and_adv_GAE(v, r, done)
print(time.time())
a.calc_realv_and_adv_GAE(v, r, done)
print(time.time())
