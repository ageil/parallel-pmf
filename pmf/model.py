import numpy as np
from collections import defaultdict

class PMF(object):
    def __init__(self, data, K, eta=(1.0, 1.0)):
        self.data = data
        self.K = K

        self.users = np.unique(data[:, 0])
        self.items = np.unique(data[:, 1])
        self.losses = []

        # initialize priors
        self.eta_theta, self.eta_beta = eta
        self.theta = defaultdict(
            lambda: np.random.normal(0, self.eta_theta, size=(self.K,)))
        self.beta = defaultdict(
            lambda: np.random.normal(0, self.eta_beta, size=(self.K,)))

    def norm_pdf(self, x, loc=0.0, scale=1.0):
        """Evaluate normal PDF at x"""
        norm = np.linalg.norm(x - loc, axis=0)
        prob = 1 / (np.sqrt(2 * np.pi) * scale)
        prob *= np.exp(-1/2 * (norm**2 / scale**2))
        return prob

    def log_norm_pdf(self, x, loc=0.0, scale=1.0):
        """Evaluate log normal PDF at x"""
        norm = np.linalg.norm(x - loc, axis=0) if type(x) == np.ndarray else x - loc
        log_prob = -np.log(scale)
        log_prob -= 1/2 * np.log(2 * np.pi)
        log_prob -= 1/2 * (norm**2 / scale**2)
        return log_prob

    def grad_log_norm_pdf(self, x, loc=0.0, scale=1.0):
        """Evaluate d/dx of the log normal PDF at x"""
        norm = np.linalg.norm(x - loc, axis=0)
        return -norm/(scale**2)

    def norm_vectors(self, x):
        """Normalize dict of vectors to unit length"""
        for i, v in x.items():
            x[i] = v / np.linalg.norm(v)
        return x

    def loss(self, data):
        """Calculate the log probability of the data under the current model"""
        loss = 0.
        for i in self.theta.keys():
            loss += self.log_norm_pdf(self.theta[i])
        for j in self.beta.keys():
            loss += self.log_norm_pdf(self.beta[j])
        for i, j, r in data:
            r_hat = self.theta[i].dot(self.beta[j])
            loss += self.log_norm_pdf(r, loc=r_hat)
        return loss

    def train(self, iters, gamma, tol=1e-4):
        """Fit the model over iters iterations with learning rate gamma"""
        # randomly initialize theta_i, beta_j (Gaussian priors)
        for i in self.users:
            self.theta[i]
        for j in self.items:
            self.beta[j]

        # normalize vectors
        self.theta = self.norm_vectors(self.theta)
        self.beta = self.norm_vectors(self.beta)

        # iteratively update latent parameters
        for iter in range(iters):
            print(iter)

            # update user preferences
            for i in self.users:
                grad = -(1 / self.eta_theta) * self.theta[i]

                # items rated by user i
                user_idx = self.data[:, 0] == i
                j_items = self.data[user_idx, 1]
                j_ratings = self.data[user_idx, 2]
                for j, rating in zip(j_items, j_ratings):
                    rating_hat = self.theta[i].dot(self.beta[j])
                    grad += (rating - rating_hat) * self.beta[j]

                update = self.theta[i] + gamma * grad
                self.theta[i] = update / np.linalg.norm(update)  # unit vector

            # update item attributes
            for j in self.items:
                # update item attributes
                grad = -(1 / self.eta_beta) * self.beta[j]

                # users rating item j
                item_idx = self.data[:, 1] == j
                i_users = self.data[item_idx, 0]
                i_ratings = self.data[item_idx, 2]
                for i, rating in zip(i_users, i_ratings):
                    rating_hat = self.theta[i].dot(self.beta[j])
                    grad += (rating - rating_hat) * self.theta[i]

                update = self.beta[j] + gamma * grad
                self.beta[j] = update / np.linalg.norm(update)  # unit vector

            # compute loss
            if iter % 10 == 0:
                loss = self.loss(self.data)
                self.losses.append(loss)
                print(loss)

        return self.losses, self.theta, self.beta

    def predict(self, data):
        """Predict ratings for users i on items j"""
        preds = np.empty((len(data),), dtype=float)
        for idx, (i, j) in enumerate(data):
            preds[idx] = self.theta[i].dot(self.beta[j])
        return preds

    def recommend(self, i):
        """get (all) recommendations for user i in sorted order"""
        items = np.array(list(self.beta.keys()))
        user = np.ones((items.shape[0],)) * i
        data = np.column_stack((user, items))
        preds = self.predict(data)
        idx = np.argsort(-preds)
        return items[idx]
