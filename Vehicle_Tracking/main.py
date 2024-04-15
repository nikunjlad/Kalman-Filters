from re import M
import cv2, argparse, torch
from requests import post
import numpy as np
from tracker import Tracker
from utils import colors_list, PutText
from yolov3.yolov3 import YOLOv3NET
from yolov3.load_weights import load_weights
from yolov3.utils.utils import post_processing, load_json



def main(args):

    # Setting and Configuring initial parameters for detections
    configs = load_json(args["detector_config"])    # reading the YOLOV3 configuration parameters
    num_classes = configs["num_classes"]            # read the number of classes
    yolo_dim = configs["im_size"]
    output_dim = args["out_dim"]
    conf_thresh = args["conf_thresh"]
    nms_thresh = args["nms_thresh"]
    weights_path = args["weights_path"]

    # Configuring the Kalman Filter Tracker
    tracker_params = load_json(args["tracker_params"])
    tracker = Tracker(tracker_params)

    # Configure GPU device for Pytorch Inference
    model = YOLOv3NET(args["detector_config"])
    model = load_weights(model, weights_path)
    device = torch.device(f"cuda:{args['device']}" if torch.cuda.is_available() and args["device"] != "cpu" else "cpu")
    #model.to(device)
    cuda = False

    # Read the video and start tracking
    frame_num = 0
    try:
        # open the video file for reading
        cap = cv2.VideoCapture(args["video_path"])

        if not cap.isOpened():
            raise Exception("Failed to open the videof file")

        while True:
            ret, frame = cap.read()     # get a frame from the capture device

            if not ret:
                break

            print(f"Frame: {frame_num}")

            # resize the frame to the expected output display
            outframe = cv2.resize(frame, output_dim)

            # resize the frame to the size of the YOLOV3 Input
            image = cv2.resize(frame, yolo_dim)
            image = torch.from_numpy(image.astype("float32")).permute(2,0,1) / 255.0

            # Add a batch dimension to the tensor
            image = image.unsqueeze(0)
            print(image.shape)
            #image = image.to(device)

            with torch.no_grad():
                model.eval()
                outputs = model(x=image, CUDA=cuda)

            print(outputs.shape)
            detections = post_processing(outputs, conf_thresh, num_classes, nms_conf=nms_thresh)
            #detections = detections.cpu()
            detections = np.array(detections)

            # Select object types to track
            to_filter = [2,5,7]
            filters = np.logical_or.reduce([detections[:, 6] == v for v in to_filter])
            detections = detections[filters]

            pred_box = detections[:,0:4]    # contains bbox information
            print(pred_box.shape)
            info_box = detections[:,4:7]    # contains confidence score, objectness score and class value
            #print(type(info_box))
            dets_all = {"dets": pred_box, "info": info_box}

            # Starting track management
            valid_detections = tracker.manage_tracks(dets_all)

            for track in tracker.tracks:
                if len(track.trace) > 0 and track.num_lost_dets <= 1:
                    t_id = track.track_id
                    boxes = track.trace[0][:4]

                    # Find the scale size between the display and YOLOV3 input dimensions
                    rw = outframe.shape[1] / yolo_dim[1]
                    rh = outframe.shape[0] / yolo_dim[0]

                    # Transform bounding box co-ordinates to output display
                    x1y1 = (int(boxes[0] * rw), int(boxes[1] * rh))
                    x2y2 = (int(boxes[2] * rw), int(boxes[3] * rh))

                    outframe = cv2.rectangle(outframe, x1y1, x2y2, colors_list[int(t_id)], 2)

                    txt = f"Id:{str(int(t_id))}"

                    outframe = PutText(outframe, text=txt, pos=x1y1, text_color=(255,255,255), bg_color=colors_list[int(t_id)],
                                       scale=0.5, thickness=1, margin=2, transparent=True, alpha=0.5)

                    for k in range(len(track.trace)):
                        x = int(track.trace[k][0] * rw) + int((int(track.trace[k][2] * rw) - int(track.trace[k][0] * rw)) / 2)
                        y = int(track.trace[k][1] * rh) + int((int(track.trace[k][3] * rh) - int(track.trace[k][1] * rh)) / 2)

                        cv2.circle(outframe, (x,y), 3, colors_list[t_id % len(colors_list)], -1)


            outframe = PutText(outframe, text=f"Frame:{frame_num}", pos=(20,40), text_color=(15,15,255), bg_color=(255,255,255),
                               scale=0.5, thickness=1, margin=3, transparent=True, alpha=0.8)
            frame_num += 1

            cv2.imshow("Vehicle Tracking", outframe)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        cap.release()
        print('MOT have been performed successfully')

if __name__ == "__main__":

    ag = argparse.ArgumentParser(description="MOT System Using YOLOV3")
    ag.add_argument("--weights-path", type=str, default="./yolov3/weights/yolov3.weights", help="weights path")
    ag.add_argument("--out-dim", type=tuple, default=(860,640), help="A tuple of expected output dimension")
    ag.add_argument("--device", default=0, help="GPU number (0,1), or 'cpu'")
    ag.add_argument("--detector-config", type=str, default="./yolov3/data/config.json")
    ag.add_argument("--tracker-params", type=str, default="./tracker_params.json")
    ag.add_argument("--conf-thresh", type=float, default=0.45, help="confidence threshold")
    ag.add_argument("--nms-thresh", type=float, default=0.5, help="NMS IOU threshold")
    ag.add_argument("--video-path", type=str, default="./videos/video2.mp4")
    args = vars(ag.parse_args())

    main(args)

