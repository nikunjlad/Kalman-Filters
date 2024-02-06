import numpy as np
from kalman_filter_1d import KalmanFilter1D
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import json, argparse

def load_json(path):
    with open(path, "r") as f:
        config = json.load(f)

    return config

def trajectory(duration, step):

    t = np.arange(0, duration, step)

    Y = 0.01 * (t**2 - t)

    return Y, t

def main(args):

    # Define the true trajectory with duration of 50 secs and step size of 0.1 secs
    Y, t = trajectory(50, 0.1)  # this initializes our 1D system

    # Load the parameters from JSON file
    tracker_params = load_json(args["tracker_params"])

    # Create Kalman Filter object
    KF = KalmanFilter1D(tracker_params)

    # Define lists for storing prediction, measurement and estimated positions
    meas_pos = []
    pred_pos = []
    est_pos = []

    # Define the std of the normal distribution used to create the measurement noise
    noise_std = 1

    for y in Y:

        predicted_state = KF.predict()
        pred_pos.append(predicted_state[0])

        # add noise to simulate the measurement process
        y = y + np.random.normal(0, noise_std)
        meas_pos.append(y)

        update_state = KF.update(y)
        est_pos.append(update_state[0])


    fig = plt.figure()
    fig.suptitle("Tracking Object in 1D space with Kalman Filter\n \Positions (m) vs Times (s)", fontsize=15, weight="bold")

    plt.plot(t, meas_pos, label="Measurement", color='blue', linewidth=0.5)
    plt.plot(t, est_pos, label="Estimated (KF)", color='red', linewidth=1)
    plt.plot(t, Y, label="True Trajectory", color='yellow', linewidth=1)

    plt.xlabel('Times (s)', fontsize=15)
    plt.ylabel('Positions (m)', fontsize=15)
    plt.legend(fontsize=15)

    mse_est = mean_squared_error(Y, est_pos)
    mse_meas = mean_squared_error(Y, meas_pos)

    print(f"Estimated (KF) MSE: {round(mse_est, 3)} m")
    print(f"Measurement MSE: {round(mse_meas, 3)} m")
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='1D Kalman Filter')
    parser.add_argument('-tp', '--tracker-params', type=str, default='tracker_params.json')
    args = vars(parser.parse_args())

    main(args)



