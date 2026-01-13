import numpy as np
from sklearn.covariance import LedoitWolf

class GaussianNormalityModel:
    """
    Gaussian normality model using Mahalanobis distance
    """

    def __init__(self, reg_param=1e-6):
        self.mean_ = None
        self.cov_inv_ = None
        self.reg_param = reg_param

    def fit(self, X):
        # Safety checks
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("NaN or Inf detected in training data")

        self.mean_ = X.mean(axis=0)

        lw = LedoitWolf()
        cov = lw.fit(X).covariance_

        # Regularization
        cov += self.reg_param * np.eye(cov.shape[0])

        try:
            self.cov_inv_ = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self.cov_inv_ = np.linalg.pinv(cov)

        print(f"✓ Model trained | samples={X.shape[0]}, features={X.shape[1]}")

    def score(self, X):
        diff = X - self.mean_
        return np.sqrt(np.sum(diff @ self.cov_inv_ * diff, axis=1))
