import numpy as np
from scipy.stats import norm


class LinearDynamicSystem:
    def __init__(self, gamma, sigma):
        self.gamma = gamma
        self.sigma = sigma

        self.log_p_x_history = []
        self.pred_mu_history = []
        self.mu_history = []

    def fit(self, observations, mu, p):

        _mu, _p = mu, p
        for i, obs in enumerate(observations):
            pred_mu, pred_p = self._predict(_mu, _p)
            _mu, _p, log_p_x = self._update(obs, pred_mu, pred_p)
            self.pred_mu_history.append(pred_mu)
            self.mu_history.append(_mu)
            self.log_p_x_history.append(log_p_x)

            print("Answer {}: ".format(i+1))

            if i > 0:
                print("predicted p.d.f. of latent variable = {:.3f}, {:.3f}"
                      .format(pred_mu, np.sum(self.log_p_x_history)))

            print("log-scaled likelihood = {:.3f}, "
                  "updated p.d.f. of latent value = {:.3f}:"
                  .format(log_p_x, _mu))

    def _predict(self, mu, p):
        pred_mu = mu
        pred_p = p + self.gamma
        return pred_mu, pred_p

    def _kalman_gain(self, p, s):
        return p / (p + s)

    def _update(self, obs, mu, p):
        k = self._kalman_gain(p, self.sigma)
        new_mu = mu + k * (obs - mu)
        new_p = (1 - k) * p
        log_p_x = norm.logpdf(x=obs, loc=new_mu, scale=np.sqrt(new_p))
        return new_mu, new_p, log_p_x


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

lds_model = LinearDynamicSystem(INIT_GAMMA, INIT_SIGMA)
lds_model.fit(INPUT_X, INIT_MU, INIT_P)
