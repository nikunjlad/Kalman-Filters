import cv2, json, argparse, sys, os
from detector import detect
from kalman_filter_2d import KalmanFilter2D
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

def load_json(path):

    with open(path, "r") as f:
        config = json.load(f)

    return config

def main(args):

    # load parameters from JSON
    kalman_params = load_json(args["tracker_params"])

    # load object positions
    GT = np.load(args["gt_path"])

    meas=[]
    pred=[]
    est=[]

    # set to 0 to avoid adding additional noise
    noise_std = 0

    track = False

    try:

        video_cap = cv2.VideoCapture(args["video_path"])

        if not video_cap.isOpened():
            raise Exception("Error: Could not open the video file")

        fr = 0

        while(True):

            # read frame
            ret, frame = video_cap.read()

            if not ret:
                break

            # detect the circle using detect function
            center = detect(frame)[0].reshape((2,1))

            if not track:

                KF = KalmanFilter2D(kalman_params, center)
                track = True

            # if we detect the circle, then track it
            if(len(center) > 0):

                # get the measurement positions
                x_meas = center[0] + np.random.normal(0, noise_std)
                y_meas = center[1] + np.random.normal(0, noise_std)
                meas.append((x_meas, y_meas))

                # obtain the groud truth for x and y co-ordinates of tracked circle
                gt = GT[fr]
                x_tp = gt[0]
                y_tp = gt[1]

                # predict next position based on newly updated position
                predicted_state = KF.predict()
                x_pred, y_pred = predicted_state[0], predicted_state[1]
                pred.append((x_pred, y_pred))

                # update state positions based on prediction and new measurement
                updated_state = KF.update(np.array([x_meas, y_meas]))
                x_est, y_est = updated_state[0], updated_state[1]
                est.append((x_est, y_est))

                # draw a rectangle for measured position
                cv2.rectangle(frame, (int(x_meas[0] - 15), int(y_meas[0] - 15)), (int(x_meas[0] + 15), int(y_meas[0] + 15)), (0,0,255), 2)

                # draw a circle for GT position
                cv2.circle(frame, (int(x_tp), int(y_tp)), 10, (0,255,255), 2)

                # draw a rectangle for estimated position
                cv2.rectangle(frame, (int(x_est[0] - 15), int(y_est[0] - 15)), (int(x_est[0] + 15), int(y_est[0] + 15)), (255,0,0), 2)

                # add legends
                cv2.rectangle(frame, (10, 30), (30, 50), (0, 0, 255), 2)
                cv2.putText(frame, "Measured Position", (40, 45), 0, 0.5, (0, 0, 0), 1)

                cv2.rectangle(frame, (10, 60), (30, 80), (255, 0, 0), 2)
                cv2.putText(frame, "Estimated Position", (40, 75), 0, 0.5, (0, 0, 0), 1)

                cv2.circle(frame, (20, 100), 10, (0, 255, 255), 2)
                cv2.putText(frame, "True Position", (40, 105), 0, 0.5, (0, 0, 0), 1)

                cv2.putText(frame, "Frame: " + str(fr), (450, 20), 0, 0.5, (0, 0, 255), 2)

            cv2.imshow('2D Object Tracking Kalman Filter', frame)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

            fr += 1

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:

        if video_cap is not None:
            video_cap.release()

        cv2.destroyAllWindows()

    print("The circle tracking has been performed successfully")

    # Plot Tracking Performance in the x and y direction

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('2D Kalman Filter Tracking Performance \n (Frames vs Positions)', fontsize=15, weight='bold')

    ax1.invert_yaxis()
    ax2.invert_yaxis()

    t = np.arange(0, len(pred), 1)
    pred = np.array(pred)
    est = np.array(est)
    meas = np.array(meas)

    # Plot the first subplot
    ax1.plot(t, GT[:, 0], label='True x Position', color="yellow", linestyle="-", linewidth=2)
    ax1.plot(t, meas[:, 0], label='Measured x Position', color="red", linestyle="-", linewidth=1)
    ax1.plot(t, est[:, 0], label='Estimated x Position', color="blue", linestyle="-", linewidth=1)

    ax1.set_title("Comparison in the x direction", fontsize=15)
    ax1.set_xlabel('Frames', fontsize=15)
    ax1.set_ylabel('x positions (pixels)', fontsize=15)
    ax1.legend(fontsize=15)
    ax1.legend(loc='lower left')

    # Plot the second subplot
    ax2.plot(t, GT[:, 1], label='True y Position', color="yellow", linestyle="-", linewidth=2)
    ax2.plot(t, meas[:, 1], label='Measured y Position', color="red", linestyle="-", linewidth=1)
    ax2.plot(t, est[:, 1], label='Estimated y Position', color="blue", linestyle="-", linewidth=1)

    ax2.set_title("Comparison in the y direction", fontsize=15)
    ax2.set_xlabel('Frames', fontsize=15)
    ax2.set_ylabel('y positions (pixels)', fontsize=15)
    ax2.legend(fontsize=15)
    ax2.legend(loc='lower left')

    # Plot comparison in 2D space
    fig, ax3 = plt.subplots()
    ax3.set_title('Comparison of Estimated (KF), Measured and True Trajectory Positions in 2D space', fontsize=12, weight='bold')
    ax3.invert_yaxis()
    ax3.plot(GT[:, 0], GT[:, 1], label='True Position', color="yellow", linestyle="-", linewidth=2)
    ax3.plot(meas[:, 0], meas[:, 1], label='Measured Position', color="red", marker="*", markersize=4, linestyle="None", linewidth=1)
    ax3.plot(est[:, 0], est[:, 1], label='Estimated Position', color="blue", marker="o", markersize=3, linestyle="None", linewidth=2)

    ax3.set_xlabel('x positions (pixels)', fontsize=15)
    ax3.set_ylabel('y positions (pixels)', fontsize=15)
    ax3.legend(fontsize=15)
    ax3.legend(loc='lower left')

    # Calculate MSE
    mse_est_x = mean_squared_error(GT[:, 0], est[:, 0])
    mse_est_y = mean_squared_error(GT[:, 1], est[:, 1])

    mse_meas_x = mean_squared_error(GT[:, 0], meas[:, 0])
    mse_meas_y = mean_squared_error(GT[:, 1], meas[:, 1])

    print(f"mse_meas_x: {mse_meas_x}")
    print(f"mse_est_x: {mse_est_x}")
    print(f"mse_meas_y: {mse_meas_y}")
    print(f"mse_est_y: {mse_est_y}")

    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='2D Kalman Filter')
    parser.add_argument('-tp', '--tracker-params', type=str, default='tracker_params.json')
    parser.add_argument('-vp', '--video-path', type=str, default='resources/input_vid.mp4')
    parser.add_argument('-gp', '--gt-path', type=str, default='resources/input_vid_gt.npy')
    args = vars(parser.parse_args())
    main(args)

    sys.exit(0)

