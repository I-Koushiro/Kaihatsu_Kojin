"""
参考リンク：https://www.avelio.co.jp/math/wordpress/?p=605
"""
import numpy as np

class KalmanFilter(object):
    def __init__(self, system_matrix, observation_matrix,
                system_cov, observation_cov,
                initial_state_mean, initial_state_cov,):
        self.system_matrix = np.array(system_matrix)
        self.observation_matrix = np.array(observation_matrix)
        self.system_cov = np.array(system_cov)
        self.observation_cov = np.array(observation_cov)
        self.initial_state_mean = np.array(initial_state_mean)
        self.initial_state_cov = np.array(initial_state_cov)


    def filter_predict(self, current_state_mean,
        current_state_cov):
        predicted_state_mean = self.system_matrix @ current_state_mean
        predicted_state_cov = (
            self.system_matrix
            @ current_state_cov
            @self.system_matrix.T
            + self.system_cov
        )

        return (predicted_state_mean, predicted_state_cov)


    def filter_update(self, predicted_state_mean, predicted_state_cov, observation):
        kalman_gain = (
            predicted_state_cov
            @ self.observation_matrix.T
            @ np.linalg.inv(
                self.observation_matrix
                @ predicted_state_cov
                @ self.observation_matrix.T
                + self.observation_cov
            )
        )
        filtered_state_mean = (
            predicted_state_mean
            + kalman_gain
            @ (observation
            - self.observation_matrix
            @ predicted_state_mean)
        )
        filtered_state_cov = (
            predicted_state_cov
            - (kalman_gain
            @ self.observation_matrix
            @ predicted_state_cov)
        )
        return (filtered_state_mean, filtered_state_cov)


    def filter(self, observations):
        observations = np.array(observations)

        n_timesteps = len(observations)
        n_dim_state = len(self.initial_state_mean)

        predicted_state_means = np.zeros((n_timesteps, n_dim_state))
        predicted_state_covs = np.zeros((n_timesteps, n_dim_state, n_dim_state))
        filtered_state_means = np.zeros((n_timesteps, n_dim_state))
        filtered_state_covs = np.zeros((n_timesteps, n_dim_state, n_dim_state))

        for t in range(n_timesteps):
            if t == 0:
                predicted_state_means[t] = self.initial_state_mean
                predicted_state_covs[t] = self.initial_state_cov
            else:
                predicted_state_means[t], predicted_state_covs[t] = self.filter_predict(
                    filtered_state_means[t-1],
                    filtered_state_covs[t-1]
                )
            filtered_state_means[t], filtered_state_covs[t] = self.filter_update(
                predicted_state_means[t],
                predicted_state_covs[t],
                observations[t]
            )

        return (
            filtered_state_means,
            filtered_state_covs,
            predicted_state_means,
            predicted_state_covs
        )


    def predict(self, n_timesteps, observations):
        (filtered_state_means,
        filtered_state_covs,
        predicted_state_means,
        predicted_state_covs) = self.filter(observations)

        _, n_dim_state = filtered_state_means.shape

        predicted_state_means = np.zeros((n_timesteps, n_dim_state))
        predicted_state_covs = np.zeros((n_timesteps, n_dim_state, n_dim_state))

        for t in range(n_timesteps):
            if t == 0:
                predicted_state_means[t], predicted_state_covs[t] = self.filter_predict(
                    filtered_state_means[-1],
                    filtered_state_covs[-1]
                )
            else:
                predicted_state_means[t], predicted_state_covs[t] = self.filter_predict(
                    predicted_state_means[t-1],
                    predicted_state_covs[t-1]
                )

        return (predicted_state_means, predicted_state_covs)


    def smooth_update(self, filtered_state_mean, filtered_state_cov,
                        predicted_state_mean, predicted_state_cov,
                        next_smoothed_state_mean, next_smoothed_state_cov):
        kalman_smoothing_gain = (
            filtered_state_cov
            @ self.system_matrix.T
            @ np.linalg.inv(predicted_state_cov)
        )
        smoothed_state_mean = (
            filtered_state_mean
            + kalman_smoothing_gain
            @ (next_smoothed_state_mean - predicted_state_mean)
        )
        smoothed_state_cov = (
            filtered_state_cov
            + kalman_smoothing_gain
            @ (next_smoothed_state_cov - predicted_state_cov)
            @ kalman_smoothing_gain.T
        )
        return (smoothed_state_mean, smoothed_state_cov)


    def smooth(self, observations):
        (filtered_state_means,
        filtered_state_covs,
        predicted_state_means,
        predicted_state_covs) = self.filter(observations)

        n_timesteps, n_dim_state = filtered_state_means.shape

        smoothed_state_means = np.zeros((n_timesteps, n_dim_state))
        smoothed_state_covs = np.zeros((n_timesteps, n_dim_state, n_dim_state))

        smoothed_state_means[-1] = filtered_state_means[-1]
        smoothed_state_covs[-1] = filtered_state_covs[-1]

        for t in reversed(range(n_timesteps - 1)):
            smoothed_state_means[t], smoothed_state_covs[t] = self.smooth_update(
                filtered_state_means[t],
                filtered_state_covs[t],
                predicted_state_means[t+1],
                predicted_state_covs[t+1],
                smoothed_state_means[t+1],
                smoothed_state_covs[t+1]
            )

        return (smoothed_state_means, smoothed_state_covs)

