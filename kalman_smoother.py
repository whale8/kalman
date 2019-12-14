# coding: utf-8
import numpy as np
import time

def KalmanSmoother(x_obs, mu0, P0, A, Q, C, R):
	"""
	mu0: 最初の潜在変数の平均
	P0: 最初の潜在変数の分散
	A: 遷移確率の平均の線形変換
	Q: 遷移確率の分散
	C: 出力確率の平均の線形変換
	R: 出力確率の分散
	"""
	T, n_feature = x_obs.shape # (n_sample, n_feature)	
	mu = mu0
	P = P0

	x_predict = np.zeros((T, n_feature))
	x_filter = np.zeros((T, n_feature))
	P_predict = np.zeros((T, n_feature, n_feature))
	V_filter = np.zeros((T, n_feature, n_feature))
	J_filter = np.zeros((T, n_feature, n_feature))

	x_predict[0] = mu.transpose()
	x_filter[0] = mu
	P_predict[0] = P0

	S = C @ P0 @ C.transpose() + R
	K = P0 @ C.transpose() @ np.linalg.pinv(S)
	V = P0 - K @ C @ P0
	V_filter[0] = V
	J_filter[0] = V @ A.transpose() @ np.linalg.pinv(P)
	
	start = time.time()

	for t in range(1, T):
		mu_ = A @ mu
		P_ = A @ V @ A.transpose() + Q

		x_predict[t] = mu_
		P_predict[t] = P_

		S = C @ P_ @ C.transpose() + R
		K = P_ @ C.transpose() @ np.linalg.pinv(S)

		mu = mu_ + K @ (x_obs[t] - C @ mu_)
		V = P_ - K @ C @ P_
		J = V @ A.transpose() @ np.linalg.pinv(P_)

		x_filter[t] = mu
		V_filter[t] = V
		J_filter[t] = J

	x_smoother = np.zeros((T, n_feature))
	V_smoother = np.zeros((T, n_feature, n_feature))
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
	print("elapsed time: ", elapsed_time)
	
	return x_predict, P_predict, x_filter, \
		V_filter, x_smoother, V_smoother, J_filter


if __name__ == "__main__":
	pass
