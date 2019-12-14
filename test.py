# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from kalman_smoother import KalmanSmoother
from train import train

# データ作成
T = 25
mu0 = np.array([100, 100])
P0 = np.array([[10, 0],
               [0, 10]])

A = np.array([[1.001, 0.001],
              [0, 0.99]])
Q = np.array([[20, 0],
              [0, 20]])
C = np.array([[1, 0],
			  [0, 1]])
R = np.array([[20, 0],
              [0, 20]])

rvq = np.random.multivariate_normal(np.zeros(2), Q, T)
rvr = np.random.multivariate_normal(np.zeros(2), R, T)
obs = np.zeros((T, 2))
obs[0] = mu0
for i in range(1, T):
    obs[i] = A @ obs[i-1]  + rvq[i] + rvr[i]


	
x_predict, V_predict, x_filter, V_filter = \
	KalmanFilter(obs, mu0, P0, A, Q, C, R)
print(x_filter)

x_predict, V_predict, x_filter, V_filter, x_smoother, V_smoother, J_filter = \
	KalmanSmoother(obs, mu0, P0, A, Q, C, R)
print(x_smoother)


mu0, P0, A, Q, C, R = train(obs)

print("mu0: ", mu0)
print("P0: ", P0)
print("A: ", A)
print("Q: ", Q)
print("C: ", C)
print("R: ", R)

x_predict_trained, _, x_filter_trained, _, x_smoother_trained, _, _ = \
	KalmanSmoother(obs, mu0, P0, A, Q, C, R)
#print(x_smoother)
print(x_predict_trained)
print()
print(x_filter_trained)
print()
print(x_smoother_trained)
"""

fig = plt.figure(figsize=(16, 9))
ax = fig.gca()

ax.scatter(obs[:, 0], obs[:, 1], s=10, alpha=1,
		   marker="o", color='w', edgecolor='k')
ax.plot(obs[:, 0], obs[:, 1], alpha=0.5, lw=1, color='k')

ax.scatter(x_filter[:, 0], x_filter[:, 1], s=10, alpha=1,
		   marker="o", color='r')
ax.plot(x_filter[:, 0], x_filter[:, 1], alpha=0.5,
		lw=1, color='r')

ax.scatter(x_smoother[:, 0], x_smoother[:, 1], s=10, alpha=1,
		   marker="o", color='b')
ax.plot(x_smoother[:, 0], x_smoother[:, 1], alpha=0.5,
		lw=1, color='b')

plt.show()
"""
