import cv2, time, json, argparse, sys
sys.path.append("/home/nikunj/knightscope/experiments/kalman_filters/Kalman_Filter_2D")
from tracker import Tracker
import seaborn as sns
from detector import detect
import numpy as np

color_list = sns.color_palette('bright', 10)
color_list = [(int(r*255), int(g*255), int(b*255)) for (r,g,b) in color_list]

def load_json(path):
    with open(path, "r") as f:
        config = json.load(f)

    return config

def main(args):

    cap = cv2.VideoCapture(args["video_path"])              # read the video from the file
    tracker_params = load_json(args["tracker_params"])      # load the tracking parameters
    tracker = Tracker(tracker_params)                       # create a tracker object, to track objects for a single camera source

    frame_nb = 0
    while(True):

        ret, frame = cap.read()     # read a frame from the video file
        if ret:                     # if frame exists
            # perform detection on the frame and get the centers of the bbox (or circles in this case).
            # this is a list of centers, each for every object detected in the scene (multiple centers for multiple objects)
            centers = detect(frame)

            # reshape centers to have each element as a column vector
            centers = centers[:,:, np.newaxis]      # convert the (1x2) centers vector to 2x1 column vector

            tracker.manage_tracks(centers)          # manage tracks by providing it the measurement (aka detected) centers

            for track in tracker.tracks:
                if len(track.trace) > 0 and track.num_lost_dets <= 2:
                    t_id = track.track_id
                    pos = track.updated

                    x, y = int(pos[0]), int(pos[1])

                    cv2.rectangle(frame, (x - 10, y - 10), (x + 10, y + 10), color_list[t_id % len(color_list)], 1)

                    cv2.putText(frame, str(track.track_id), (x - 10, y - 20), 0, 0.5, color_list[t_id % len(color_list)], 2)

                    for k in range(len(track.trace)):
                        x = int(track.trace[k][0])
                        y = int(track.trace[k][1])

                        cv2.circle(frame, (x,y), 3, color_list[t_id % len(color_list)], -1)

            frame_nb_text = f"Frame: {frame_nb}"
            cv2.putText(frame, frame_nb_text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 1)

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_nb += 1
            time.sleep(0.02)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Multi-Object Tracking System")
    parser.add_argument('--tracker_params', type=str, default='tracker_params.json')
    parser.add_argument('--video_path', type=str, default="../resources/multi_input_vid.mp4")
    args = vars(parser.parse_args())

    main(args)

    sys.exit(0)
