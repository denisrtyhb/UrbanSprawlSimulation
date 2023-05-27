import numpy as np
from tqdm import tqdm
import sklearn.linear_model
import lib
from tqdm import tqdm

class LogisticRegression:
    _LAP: float = 0.6
    _TIP: float = 0.1
    _threshold: float = 0.3
    _neigh_radius: int = 5
    _pred: int

    map_height: int
    map_width: int

    def __init__(self, LAP=None, TIP=None, neigh_radius=None, threshold=None):
        if LAP is not None:
            self._LAP = LAP

        if TIP is not None:
            self._TIP = TIP

        if neigh_radius is not None:
            self._neigh_radius = neigh_radius

        if threshold is not None:
            self._threshold = threshold

    def change_paramgs(**kwargs):
        if 'LAP' in kwargs:
            self._LAP = kwargs['LAP']

        if 'TIP' in kwargs:
            self._LAP = kwargs['TIP']

        if 'threshold' in kwargs:
            self._threshold = kwargs['threshold']

        if 'neigh_radius' in kwargs:
            self._neigh_radius = kwargs['neigh_radius']


    def fit(self, before, after, features):

        before_vec = np.copy(before.reshape(-1))
        b = np.copy(after.reshape(-1))
        A = features.reshape(features.shape[0], -1).T
        # A[i] -> b[i]

        # only predicting on before == 0
        b[before_vec != 0] = 0
        A[before_vec != 0, :] = 0


        model = sklearn.linear_model.LogisticRegression(
            random_state=0,
            max_iter=10000,
            tol=0.01,
            verbose=0,
            solver='newton-cholesky',
            class_weight='balanced',
        ).fit(A, b)

        self._pred = model.predict_proba(A)[:,1].reshape(before.shape)


    def predict(self, array, t=1):
        p_func = lib.get_p_func(
            lambda: self._pred,
            LAP=self._LAP,
            neigh_radius=self._neigh_radius,
            TIP=self._TIP
        )
        return lib.simulate_deter(array, p_func, self._threshold, t)

class GWR:
    _LAP: float = 0.6
    _TIP: float = 0.1
    _bandwidth: float = 0.31
    _threshold: float = 0.3
    _neigh_radius: int = 5
    _pred: int

    map_height: int
    map_width: int

    _rng: np.random.default_rng

    def __init__(self, LAP=None, TIP=None, bandwidth=None, neigh_radius=None, threshold=None, seed=None):
        if LAP is not None:
            self._LAP = LAP

        if TIP is not None:
            self._TIP = TIP

        if neigh_radius is not None:
            self._neigh_radius = neigh_radius

        if bandwidth is not None:
            self._bandwidth = bandwidth

        if threshold is not None:
            self._threshold = threshold

        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng(12345)

    def change_paramgs(**kwargs):
        if 'LAP' in kwargs:
            self._LAP = kwargs['LAP']

        if 'TIP' in kwargs:
            self._LAP = kwargs['TIP']

        if 'threshold' in kwargs:
            self._threshold = kwargs['threshold']

        if 'neigh_radius' in kwargs:
            self._neigh_radius = kwargs['neigh_radius']


    def fit(self, before, after, features, samples_count=None):
        if before.shape != after.shape:
            print("fit error, arrays have different shapes")
            return
        if samples_count is None:
            samples_count = int(np.sqrt(np.sum((before == 0) + (before == 1))))
        self.map_height, self.map_width = before.shape

        def random_point():
            x = self._rng.integers(low=0, high=self.map_height, size=1)[0]
            y = self._rng.integers(low=0, high=self.map_width, size=1)[0]
            if before[x, y]:
                return random_point()
            return x, y

        def n_random_points(n: int):
            return np.array([random_point() for i in range(samples_count)])

        samples = n_random_points(samples_count)
        samples_x = samples[:,0]
        samples_y = samples[:,1]

        sample_features = features[:, samples_x, samples_y]
        sample_answers = after[samples_x, samples_y]


        xv, yv = np.meshgrid(np.arange(0, samples_count), np.arange(0, samples_count))
        dist_matrix = np.sqrt(np.linalg.norm(samples[xv] - samples[yv], axis=2, ord=2))

        def w(d):
            return np.exp(-d ** 2 / self._bandwidth ** 2)

        W = w(dist_matrix)

        model = [None] * samples_count
        X = sample_features.T
        y = sample_answers

        for i in tqdm(range(samples_count)):
            model[i] = sklearn.linear_model.LogisticRegression(
                random_state=0,
                max_iter=100000,
                tol=0.05,
                verbose=0,
                solver='lbfgs',
                # class_weight='balanced',
            ).fit(X, y, W[i])

        pred = np.zeros(before.shape)

        for i in tqdm(range(self.map_height)):
            for j in range(self.map_width):
                if before[i, j]:
                    continue
                ind = np.argmin((samples_x - i) ** 2 + (samples_y - j) ** 2)
                pred[i, j] = model[ind].predict_proba(features[:, i, j].reshape(1, -1))[0, 1]

        self._pred = pred
        changed = np.sort(pred[(before == 0) * (after == 1)])
        remained = np.sort(pred[(before == 0) * (after == 0)])

        self.changed = changed
        self.remained = remained


    def predict(self, array, t=1):
        p_func = lib.get_p_func(
            lambda: self._pred,
            LAP=self._LAP,
            neigh_radius=self._neigh_radius,
            TIP=self._TIP
        )
        return lib.simulate_deter(array, p_func, self._threshold, t)
