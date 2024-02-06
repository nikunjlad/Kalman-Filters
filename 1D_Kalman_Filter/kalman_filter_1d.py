import sys
import numpy as np

class KalmanFilter1D:

    def __init__(self, params):

        # Getting Kalman Filter parameters
        dt = params["dt"]   # the time interval
        u = params["u"]     # control input (in our case the acceleration component)
        std_meas = params["std_meas"]   # standard deviation of the measurement noise
        std_acc = params["std_acc"]     # standard deviation of the acceleration component (i.e. the control input)

        self.x = np.array([[0],
                           [0]])        # a 2x1 array of zeros for the system state (which holds the position and velocity of the object)

        self.u = u

        self.A = np.array([[1, dt],
                           [0, 1]])     # 2x2 state transition matrix to transition our state from previous time to current time

        self.B = np.array([[(dt**2)/2],
                           [dt]])           # a 2x1 control input matrix

        self.H = np.array([[1, 0]])         # a 1x2 tranformation matrix

        self.Q = np.array([[(dt**4)/4, (dt**3)/2],
                           [(dt**3)/2, dt**2]]) * std_acc**2    # a 2x2 process covariance matrix

        self.R = std_meas **2       # a scalar representing measurement noise covariance

        self.P = np.eye(self.A.shape[1]) * 0.0001   # a 2x2 diagonal error covariance matrix which is initialized with small values to act as apriori variable

    def predict(self):

        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)    # predicting the next state of our system using previous a posteriori conditions

        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q  # Projecting the error covariances using previous a posteriori conditions

        if np.isnan(self.P).any():
            print("Unable to continue tracking: The parameters are not properly tuned")
            sys.exit(1)

        return self.x

    def update(self, z):

        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))      # The Kalman Gain

        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))   # Updating the system state estimate using the new measurement

        I = np.eye(self.H.shape[1])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)        # Updating the system error covariance using new variables

        return self.x