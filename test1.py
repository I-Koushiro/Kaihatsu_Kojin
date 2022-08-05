import Kalman
import numpy as np
import matplotlib.pyplot as plt

n_timesteps = 120
np.random.seed(1)

system_matrix = [[1][1][1]]
observation_matrix = [[1][1][1]]
system_cov = [[1, 1, 1]
              [1, 1, 1]
              [1, 1, 1]]
observation_cov = [[1, 1, 1]
                   [1, 1, 1]
                   [1, 1, 1]]
initial_state_mean = [[0][0][0]]
initial_state_cov = [[0, 0, 0]
                     [0, 0, 0]
                     [0, 0, 0]]

states = np.zeros(3, n_timesteps)
observations = np.zeros(3, n_timesteps)
# np.random.multivariate_normal(mean, cov, size, check_valid, tol)
# mean:生成したい正規分布の平均値ベクトル
# cov:生成したい正規分布の分散共分散行列
# size:生成する乱数の数
system_noise = np.random.multivariate_normal(
    np.zeros(3),
    system_cov,
    n_timesteps
)
observation_noise = np.random.multivariate_normal(
    np.zeros(3),
    observation_cov,
    n_timesteps
)

states[0][0] = initial_state_mean[0][0] + system_noise[0][0]
states[0][1] = initial_state_mean[1][0] + system_noise[1][0]
states[0][2] = initial_state_mean[2][0] + system_noise[2][0]

observations[0][0] = states[0][0] + observation_noise[0][0]
observations[0][1] = states[0][1] + observation_noise[1][0]
observations[0][2] = states[0][2] + observation_noise[2][0]
