import numpy as np
from numpy import log
from numpy.random import normal


class LinearDynamicSystem:
    def __init__(self, p, mu, gamma, sigma):
        self.p = p
        self.x = mu
        self.gamma = gamma
        self.sigma = sigma
        self.xs = []

    def fit(self, observations):
        for i, obs in enumerate(observations):
            self.z = normal(loc=self.x, scale=self.p)

            if i > 0:
                print("predicted p.d.f. of latent variable = {:.3f}, {:.3f}"
                      .format(self.z, log(np.prod(self.xs))))

            pred_z, pred_x, pred_p = self._perform_expectation(self.z)
            self.p = pred_p

            x = self._perform_maximization(pred_x, obs, pred_z)
            self.xs.append(x)

            print("Answer {}: ".format(i+1))
            print("log-scaled likelihood = {:.3f}, "
                  "updated p.d.f. of latent value = {:.3f}:"
                  .format(log(x), pred_z))

    def _perform_expectation(self, z):
        pred_z = z + self._gen_noize(self.gamma)
        pred_x = pred_z + self._gen_noize(self.sigma)
        pred_p = self.p + self.gamma
        return pred_z, pred_x, pred_p

    def _gen_noize(self, var):
        return normal(loc=0, scale=var)

    def _perform_maximization(self, x, obs, mu):
        k = self.p / (self.p + self.sigma)
        x = mu + k * (obs - x)
        return x


def generate_observation(b):
    return 20 * b + 10


STUDENT_ID = input("please, input a your id:")
print("Your ID is {}".format(STUDENT_ID))

b_1, b_2, b_3, b_4 = [int(sid) for sid in STUDENT_ID[-4:]]
print("b_1 = {}, b_2 = {}, b_3 = {}, b_4 = {}".format(b_1, b_2, b_3, b_4))

INPUT_X = np.array([generate_observation(b) for b in [b_1, b_2, b_3, b_4]])
INIT_P = 50
INIT_MU = 100
INIT_GAMMA = 10
INIT_SIGMA = 20
print("initialized value...: X = {}, P_0 = {}, μ_0 = {}, Γ = {}, Σ = {}"
      .format(INPUT_X, INIT_P, INIT_MU, INIT_GAMMA, INIT_SIGMA))


lds_model = LinearDynamicSystem(INIT_P, INIT_MU, INIT_GAMMA, INIT_SIGMA)
lds_model.fit(INPUT_X)
