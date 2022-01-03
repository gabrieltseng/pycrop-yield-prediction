import numpy as np
from scipy.spatial.distance import pdist, squareform


class GaussianProcess:
    """
    The crop yield Gaussian process
    """

    def __init__(self, sigma=1, r_loc=0.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01):
        self.sigma = sigma
        self.r_loc = r_loc
        self.r_year = r_year
        self.sigma_e = sigma_e
        self.sigma_b = sigma_b

    @staticmethod
    def _normalize(x):
        x_mean = np.mean(x, axis=0, keepdims=True)
        x_scale = np.ptp(x, axis=0, keepdims=True)

        return (x - x_mean) / x_scale

    def run(
        self,
        feat_train,
        feat_test,
        loc_train,
        loc_test,
        year_train,
        year_test,
        train_yield,
        model_weights,
        model_bias,
    ):

        # makes sure the features have an additional testue for the bias term
        # We call the features H since the features are used as the basis functions h(x)
        H_train = np.concatenate(
            (feat_train, np.ones((feat_train.shape[0], 1))), axis=1
        )
        H_test = np.concatenate((feat_test, np.ones((feat_test.shape[0], 1))), axis=1)

        Y_train = np.expand_dims(train_yield, axis=1)

        n_train = feat_train.shape[0]
        n_test = feat_test.shape[0]

        locations = self._normalize(np.concatenate((loc_train, loc_test), axis=0))
        years = self._normalize(np.concatenate((year_train, year_test), axis=0))
        # to calculate the se_kernel, a dim=2 array must be passed
        years = np.expand_dims(years, axis=1)

        # These are the squared exponential kernel function we'll use for the covariance
        se_loc = squareform(pdist(locations, "euclidean")) ** 2 / (self.r_loc ** 2)
        se_year = squareform(pdist(years, "euclidean")) ** 2 / (self.r_year ** 2)

        # make the dirac matrix we'll add onto the kernel function
        noise = np.zeros([n_train + n_test, n_train + n_test])
        noise[0:n_train, 0:n_train] += (self.sigma_e ** 2) * np.identity(n_train)

        kernel = ((self.sigma ** 2) * np.exp(-se_loc) * np.exp(-se_year)) + noise

        # since B is diagonal, and B = self.sigma_b * np.identity(feat_train.shape[1]),
        # its easy to calculate the inverse of B
        B_inv = np.identity(H_train.shape[1]) / self.sigma_b
        # "We choose b as the weight vector of the last layer of our deep models"
        b = np.concatenate(
            (model_weights.transpose(1, 0), np.expand_dims(model_bias, 1))
        )

        K_inv = np.linalg.inv(kernel[0:n_train, 0:n_train])

        # The definition of beta comes from equation 2.41 in Rasmussen (2006)
        beta = np.linalg.inv(B_inv + H_train.T.dot(K_inv).dot(H_train)).dot(
            H_train.T.dot(K_inv).dot(Y_train) + B_inv.dot(b)
        )

        # We take the mean of g(X*) as our prediction, also from equation 2.41
        pred = H_test.dot(beta) + kernel[n_train:, :n_train].dot(K_inv).dot(
            Y_train - H_train.dot(beta)
        )

        return pred
