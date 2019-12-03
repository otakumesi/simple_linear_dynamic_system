import numpy as np
from numpy import log
from numpy.random import normal


class LinearDynamicSystem:
    def __init__(self, p, mu, gamma, sigma):
        self.p = p
        self.x = mu
        self.H = 1
        self.gamma = gamma
        self.sigma = sigma
        self.x_history = []

    def fit(self, observations):

        for i, obs in enumerate(observations):
            pred_z, pred_p = self._perform_expectation(self.x)

            new_x, new_p = self._perform_maximization(obs, pred_z, pred_p)
            self.x = new_x
            self.p = new_p

            self.x_history.append(new_x)

            print("Answer {}: ".format(i+1))
            print("log-scaled likelihood = {:.3f}, "
                  "updated p.d.f. of latent value = {:.3f}:"
                  .format(log(new_x), pred_z))

            if i > 0:
                print("predicted p.d.f. of latent variable = {:.3f}, {:.3f}"
                      .format(self.x, log(np.prod(self.x_history))))



    def _perform_expectation(self, x):
        pred_z = x
        pred_p = self.p + self.gamma
        return pred_z, pred_p

    def _karman_gain(self, p, s):
        return p / (p + s)

    def _gen_noize(self, var):
        return normal(loc=0, scale=var)

    def _perform_maximization(self, obs, z, p):
        k = self._karman_gain(p, self.sigma)
        new_x = z + k * (obs - z)
        new_p = (1 - k) * p
        return new_x, new_p


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
