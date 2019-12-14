# coding: utf-8
import numpy as np
import time

def KalmanFilter(x_obs, mu0, P0, A, Q, C, R):
	"""
	mu0: 最初の潜在変数の平均
	P0: 最初の潜在変数の分散
	A: 遷移確率の平均の線形変換
	Q: 遷移確率の分散
	C: 出力確率の平均の線形変換
	R: 出力確率の分散
	"""
	T, n_feature = x_obs.shape
	mu = mu0
	P = P0

	x_predict = np.zeros((T, n_feature))
	x_filter = np.zeros((T, n_feature))
	P_predict = np.zeros((T, n_feature, n_feature))
	V_filter = np.zeros((T, n_feature, n_feature))

	x_predict[0] = mu.transpose()
	x_filter[0] = mu
	P_predict[0] = P
	V_filter[0] = P

	start = time.time()

	for t in range(1, T):
		mu_ = A @ mu
		P_ = A @ P @ A.transpose() + Q

		x_predict[t] = mu_
		P_predict[t] = P_

		S = C @ P_ @ C.transpose() + R
		K = P_ @ C.transpose() @ np.linalg.inv(S)

		mu = mu_ + K @ (x_obs[t] - mu_)
		V = P_ - K @ C @ P_

		x_filter[t] = mu
		V_filter[t] = V
		
	elapsed_time = time.time() - start
	print("elapsed time: ", elapsed_time)

	return x_predict, P_predict, x_filter, V_filter
