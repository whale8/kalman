# coding: utf-8
import numpy as np
import time

def KalmanSmoother(T, x_obs, mu0, P0, A, b, Q, C, R):
	"""
	mu0: 最初の潜在変数の平均
	P0: 最初の潜在変数の分散
	A: 遷移確率の平均の線形変換
	Q: 遷移確率の分散
	C: 出力確率の平均の線形変換
	R: 出力確率の分散
	"""
	
	mu = mu0
	P = P0

	x_predict = np.zeros((T, 2))
	x_filter = np.zeros((T, 2))
	P_predict = np.zeros((T, 2, 2))
	V_filter = np.zeros((T, 2, 2))
	J_filter = np.zeros((T, 2, 2))

	x_predict[0] = mu.transpose()
	x_filter[0] = mu
	P_predict[0] = P
	V_filter[0] = P
	J_filter[0] = P @ A.transpose() @ np.linalg.inv(P)

	start = time.time()

	for t in range(1, T):
		mu_ = A @ mu + b
		P_ = A @ P @ A.transpose() + Q

		x_predict[t] = mu_
		P_predict[t] = P_

		S = C @ P_ @ C.transpose() + R
		K = P_ @ C.transpose() @ np.linalg.inv(S)

		mu = mu_ + K @ (x_obs[t] - mu_)
		V = P_ - K @ C @ P_
		J = V @ A.transpose() @ np.linalg.inv(P_)

		x_filter[t] = mu
		V_filter[t] = V
		J_filter[t] = J

	x_smoother = np.zeros((T, 2))
	V_smoother = np.zeros((T, 2, 2))
	x_smoother[-1, :] = x_filter[-1, :]
	V_smoother[-1, :, :] = V_filter[-1, :, :]
	
	for t in reversed(range(0, T-1)):
		mu_ = x_smoother[t+1] - A @ x_filter[t] 
		mu = x_filter[t] + J_filter[t] @ mu_
		V_ = V_smoother[t+1] - P_predict[t]
		V = V_filter[t] + J_filter[t] @ V_ @ J_filter[t].transpose()

		x_smoother[t] = mu
		V_smoother[t] = V

	elapsed_time = time.time() - start
	print("numpy: ", elapsed_time)
	
	return x_predict, P_predict, x_filter, \
		V_filter, x_smoother, V_smoother


if __name__ == "__main__":
	pass
