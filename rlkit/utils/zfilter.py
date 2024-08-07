import numpy as np
import pdb
# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape).astype(np.float32)
        self._S = np.zeros(shape).astype(np.float32)

    def push(self, x):
        x = np.asarray(x).astype(np.float32)

        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, conditioner = 1.0, demean=True, destd=True, clip=5.0):
        self.conditioner = conditioner
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update):
        x = x * self.conditioner
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x.astype(np.float32)

