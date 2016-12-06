import numpy as np
from matplotlib import pyplot as plt

plt.style.use('ggplot')

## ----------------------------------------------------------------------
## MOTION MODELS

class step_1d:
	# linear motion in 1d world, step in one of two possible directions
	# p_right: probability of moving right
	# noise_var: variance of noise
	def __init__(self, step_size, p_right, noise_var):
		self.step_size = step_size
		self.p_right = p_right
		self.noise_var = noise_var

	def move(self, state):
		# return next state, expected motion and noise covariance
		dim = state.shape[0]
		predicted_motion = np.zeros(state.shape)
		predicted_motion[0] = self.step_size * (2 * (np.random.rand() < self.p_right) - 1)
		noise = np.zeros(state.shape)
		noise[0] = np.sqrt(self.noise_var)*np.random.randn()
		next_state = state + predicted_motion + noise
		noise_cov = np.zeros([dim,dim])
		noise_cov[0,0] = self.noise_var

		return next_state, predicted_motion, noise_cov


## ----------------------------------------------------------------------
## MEASUREMENT MODELS

class distance_estimation_1d:
	# estimate distance to each landmark
	def __init__(self, noise_var):
		self.noise_var = noise_var

	def gradient(self, state, landmark):
		grad = np.zeros(state.shape)
		grad[0] = -1
		grad[landmark] = 1
		return grad

	def exact_measure(self, state, landmark):
		# return exact distance between agent and landmark
		return np.array([state[landmark] - state[0]])

	def measure(self, state, landmark):
		# return estimated distance between agent and landmark, 
		#        measurement gradient and noise covariance
		# state: current state
		# landmark: observed landmark
		dist = self.exact_measure(state, landmark) + self.noise_var*np.random.randn()
		return dist, self.gradient(state, landmark), np.array([[self.noise_var]])

## ----------------------------------------------------------------------
## MOTION UPDATE

def EIF_estimate(H, b):
	# H: information matrix; b: information vector
	# return EIF estimate of state vector
	return np.dot(b, np.linalg.inv(H)).T

def motion_update(H, b, predicted_motion, noise_cov):
	# motion update for information matrix and information vector
	mean_state = EIF_estimate(H, b)
	H_next = np.linalg.inv(np.linalg.inv(H) + noise_cov)
	new_mean_state = mean_state + predicted_motion
	b_next = np.dot(new_mean_state.T, H_next)
	return H_next, b_next

def linear_measurement_update(H, b, measurement_model, state, landmark):
	# measurement update for linear observation function
	z, grad, noise_cov = measurement_model.measure(state, landmark)
	H_new = H + np.dot(grad, np.dot(np.linalg.inv(noise_cov), grad.T))
	b_new = b + np.dot(z.T, np.dot(np.linalg.inv(noise_cov), grad.T))
	return H_new, b_new

if __name__ == '__main__':
	motion_model = step_1d(2, 0.5, 2.0)
	measurement_model = distance_estimation_1d(2.0)

	# initialization
	state = np.array([[0,0]]).T
	H = np.identity(state.shape[0])
	b = np.zeros(state.shape).T

	# simulation
	T = 1000 # number of time steps
	states = np.zeros([state.shape[0], T]) # real states
	states[:,0] = np.squeeze(state)
	estimation_filter = np.zeros([state.shape[0], T]) # EIF estimations
	estimation_simple = np.zeros([state.shape[0], T]) # simple mean estimations
	for t in range(1,T):
	    state, predicted_motion, noise_cov = motion_model.move(state)
	    states[:,t] = np.squeeze(state)
	    estimation_simple[:,t] = estimation_simple[:,t-1] + np.squeeze(predicted_motion)
	    H, b = motion_update(H, b, predicted_motion, noise_cov)
	    for landmark in range(1, state.shape[0]):
	        H, b = linear_measurement_update(H, b, measurement_model, state, landmark)
	    estimation_filter[:,t] = np.squeeze(EIF_estimate(H,b))

	plt.figure()
	plt.plot(states[0,:])
	plt.plot(estimation_simple[0,:])
	plt.plot(estimation_filter[0,:])
	plt.legend(['Real position', 'Simple estimate', 'Filtered estimate'])
	plt.show()

