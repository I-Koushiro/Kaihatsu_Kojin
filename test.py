import Kalman
import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(1) # 乱数固定

n_timesteps = 120

system_matrix = [[1]]
observation_matrix = [[1]]
system_cov = [[1]]
observation_cov = [[10]]
initial_state_mean = [0]
initial_state_cov = [[1e7]]


states = np.zeros(n_timesteps)
observations = np.zeros(n_timesteps)
system_noise = np.random.multivariate_normal(
    np.zeros(1),
    system_cov,
    n_timesteps
)
observation_noise = np.random.multivariate_normal(
    np.zeros(1),
    observation_cov,
    n_timesteps
)

states[0] = initial_state_mean + system_noise[0]
observations[0] = states[0] + observation_noise[0]
for t in range(1, n_timesteps):
    states[t] = states[t-1] + system_noise[t]
    observations[t] = states[t] + observation_noise[t]

# 分析するデータ(時刻0から99まで)と予測と比較する用のデータ(時刻100以降)に分けておく
states_train = states[:100]
states_test = states[100:]
observations_train = observations[:100]
observations_test = observations[100:]

upper = 17
lower = -12

fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(np.arange(n_timesteps), states, label=" true state")
ax.plot(np.arange(n_timesteps), observations, color="gray", label="observation")
ax.set_title("simulation data")
ax.set_ylim(lower, upper)
ax.legend(loc='upper left')

# モデル生成
model = Kalman.KalmanFilter(
    system_matrix,
    observation_matrix,
    system_cov,
    observation_cov,
    initial_state_mean,
    initial_state_cov
)

# フィルタリング分布取得
filtered_state_means, filtered_state_covs, _, _ = model.filter(observations_train)

# 予測分布取得
predicted_state_means, predicted_state_covs = model.predict(len(states_test), observations_train)

# 平滑化分布取得
smoothed_state_means, smoothed_state_covs = model.smooth(observations_train)

upper = 17
lower = -10

T1 = len(states_train)
T2 = len(states_test)

# フィルタリング・予測・平滑化の下側95％点と上側95％点
filtering_lower95 = filtered_state_means.flatten() - 1.96 * filtered_state_covs.flatten()**(1/2)
filtering_upper95 = filtered_state_means.flatten() + 1.96 * filtered_state_covs.flatten()**(1/2)
prediction_lower95 = predicted_state_means.flatten() - 1.96 * predicted_state_covs.flatten()**(1/2)
prediction_upper95 = predicted_state_means.flatten() + 1.96 * predicted_state_covs.flatten()**(1/2)
smoothing_lower95 = smoothed_state_means.flatten() - 1.96 * smoothed_state_covs.flatten()**(1/2)
smoothing_upper95 = smoothed_state_means.flatten() + 1.96 * smoothed_state_covs.flatten()**(1/2)


fig, axes = plt.subplots(nrows=3, figsize=(16, 12))

# フィルタリングと予測
axes[0].plot(np.arange(T1+T2), states, label="true state")
axes[0].plot(np.arange(T1), filtered_state_means.flatten(), color="red", label="filtering")
axes[0].fill_between(np.arange(T1), filtering_lower95, filtering_upper95, color='red', alpha=0.15)
axes[0].plot(np.arange(T1, T1+T2), predicted_state_means.flatten(), color="indigo", label="prediction")
axes[0].fill_between(np.arange(T1, T1+T2), prediction_lower95, prediction_upper95, color="indigo", alpha=0.15)
axes[0].axvline(100, color="black", linestyle="--", alpha=0.5)
axes[0].set_ylim(lower, upper)
axes[0].legend(loc='upper left')
axes[0].set_title("Filtering & Prediction")

# 平滑化
axes[1].plot(np.arange(T1+T2), states, label="true state")
axes[1].plot(np.arange(T1), smoothed_state_means.flatten(), color="darkgreen", label="smoothing")
axes[1].fill_between(np.arange(T1), smoothing_lower95, smoothing_upper95, color='darkgreen', alpha=0.15)
axes[1].axvline(100, color="black", linestyle="--", alpha=0.5)
axes[1].set_ylim(lower, upper)
axes[1].legend(loc='upper left')
axes[1].set_title("Smoothing")

# フィルタリングと平滑化の信頼区間の比較
axes[2].plot(np.arange(T1), filtering_lower95, color='red', linestyle="--", label='filtering confidence interval')
axes[2].plot(np.arange(T1), filtering_upper95, color='red', linestyle="--")
axes[2].fill_between(np.arange(T1), smoothing_lower95, smoothing_upper95, color='darkgreen', alpha=0.3, label='smoothing confidence interval')
axes[2].axvline(100, color="black", linestyle="--", alpha=0.5)
axes[2].set_xlim(-6, 125)
axes[2].set_ylim(lower, upper)
axes[2].legend(loc='upper left')
axes[2].set_title("Confidence interval: Filtering v.s. Smoothing")

plt.show()