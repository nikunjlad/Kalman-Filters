import sys
import numpy as np

class KalmanFilter2D:

    def __init__(self, params, detection):

        # Read the Kalman filter parameters
        dt = params["dt"]                           # the time interval
        u_x = u_y = params["u"]                     # the control input in X and Y direction(in our case the acceleration component)
        std_acc = params["std_acc"]            # the standard deviation of the acceleration component
        std_meas_x = params["std_meas_x"]      # the standard deviation of the measurement in X direction
        std_meas_y = params["std_meas_y"]      # the standard deviation of the measurement in Y direction

        self.u = np.array([[u_x],
                           [u_y]])      # A 2x1 vector

        self.x = np.array([[0],
                           [0],
                           [0],
                           [0]])        # A 4x1 vector to hold state information in X and Y direction for position and velocity

        self.x[0:2] = detection         # store the initial detection position information in the first 2 rows of the state variable

        self.A = np.array([[1,0,dt,0],
                           [0,1,0,dt],
                           [0,0,1,0],
                           [0,0,0,1]])      # A 4x4 State Transition Matrix to transition state from previous time to current time

        self.B = np.array([[(dt**2)/2,0],
                           [0,(dt**2)/2],
                           [dt,0],
                           [0,dt]])         # A 4x2 Control Input Matrix

        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]])      # A 2x4 Transformation Matrix

        self.Q = np.array([[(dt**4)/4,0,(dt**3)/2,0],
                           [0,(dt**4)/4,0,(dt**3)/2],
                           [(dt**3)/2,0,(dt**2),0],
                           [0,(dt**3)/2,0,(dt**2)]]) * std_acc**2      # A 4x4 Process Noise Covariance Matrix

        self.R = np.array([[std_meas_x**2, 0],
                           [0, std_meas_y**2]])     # A 2x2 Measurement Noise Covariance Matrix

        self.P = np.eye(self.A.shape[1]) * 1000     # A 4x4 Initial Covariance Matrix (to act as a priori)

    def predict(self):

        # predict the state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)        # Xₖ = Axₖ₋₁ + Buₖ₋₁

        # Calculate Error Covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q      # Pₖ = APₖ₋₁Aᵀ + Q

        return self.x[0:2]

    def update(self, z):

        # Calculate the Kalman Gain
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R       # S = HPₖHᵀ + R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))      # K = PₖHᵀ(S)⁻¹  i.e PₖHᵀ(HPₖHᵀ + R)⁻¹

        # Update the estimate with measurement z
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))   # X = xₖ + Kₖ(zₖ - Hxₖ)

        # Update the error covariance
        I = np.eye(self.H.shape[1])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)       # P = (I - HKₖ)Pₖ

        return self.x[0:2]