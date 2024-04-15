import numpy as np
from Kalman_Filter import KalmanFilter
from tracker_utils import track_association

class Track:

    def __init__(self, params, detection, dets_meta, trackId):
        self.track_id = trackId
        self.KF = KalmanFilter(params, detection)
        self.num_lost_dets = 0
        self.trace = []
        self.best_confidence = dets_meta[0]
        self.best_instance = dets_meta

    def predict(self):
        self.predicted = self.KF.predict()

    def update(self, detection):
        self.updated = self.KF.update(detection)

class Tracker():

    def __init__(self, params):
        self.params = params        # read the parameters from the JSON file
        self.min_iou = self.params["min_iou"]     # get the minimum distance the centers should have to be considered a valid candidate for association
        self.max_lost_dets = self.params["max_lost_dets"]
        self.trace_length = self.params["trace_length"]
        self.id = self.params["id"]
        self.tracks = []

    def manage_tracks(self, dets):

        valid_detections = []       # this holds the valid frames to be sent for post-processing downstream

        # unpack the detection dictionary to obtain the bbox information and the metadata of each detection (like confidence score, class info, etc)
        detections, info = dets['dets'], dets['info']

        # given new detections (or measurements), first predict the next state of the objects in space using the 2D Kalman Filter predict function
        for i in range(len(self.tracks)):
            self.tracks[i].predict()        # Update each track in track list with new predicted states

        # given the new predicted states of the object, we do track association.
        # Basically we match currently tracked objects (with the predicted states) to the new measurements
        # We create pairs between which objects match best from previous time frame to current frame. Likewise, we keep track of unpaired detections
        # and tracks.
        pairs, unpaired_dets, unpaired_tracks = track_association(self.tracks, detections, self.min_iou)

        # For Associated pairs
        # For objects whose associations were successful between predicted and measured positions in space, update the trace list
        for i,j in pairs:
            self.tracks[i].num_lost_dets = 0        # Reset the ith track lost detections counter to 0. That's because the object is still present hence not lost
            if self.tracks[i].best_confidence < info[j][0]:     # If current detection confidence is better than the one tracked previously, then update the info
                self.tracks[i].best_confidence = info[j][0]     # keeps track of the best confidence object in a particular track
                self.tracks[i].best_instance = info[j]          # keeps track of the best instance of the object based on confidence in a particular track

            # Update the ith track after association with the measurement information.
            # The updation step corrects any errors made in the Kalman Filter prediction using the incoming detection measurements
            self.tracks[i].update(detections[j])

            updated_state = self.tracks[i].updated[:4]      # extract the first
            #updated_state = np.concatenate([updated_state, info[j].reshape((-1,1))])

            # if length of trace is greater than 40 (since we are keeping a history of past 40 points the object traversed in space), remove the oldest one
            if len(self.tracks[i].trace) >= self.trace_length:
                self.tracks[i].trace = self.tracks[i].trace[:-1]    # remove oldest track

            # Insert new updated position of the object in the trace list (which the KF updated earlier).
            self.tracks[i].trace.insert(0, updated_state)

        # FOR UNPAIRED TRACKS
        # check and delete each track that has undergone lost detection for more than max number of frames
        del_track = 0
        for i in unpaired_tracks:
            if self.tracks[i - del_track].num_lost_dets > self.max_lost_dets:    # if track lost more than threshold, delete it
                print(unpaired_tracks, len(self.tracks))
                valid_detections.append(self.tracks[i - del_track].best_instance)   # add the instance to the valid detections list before deleting
                del self.tracks[i - del_track]      # delete tracked object if lost for a long time
                del_track += 1
            else:
                self.tracks[i - del_track].num_lost_dets += 1       # if object was just missed once, then increment counter

        # FOR UNPAIRED DETECTIONS
        # This loop is triggered when new detections come and they are not yet assigned a track
        # This loop is also triggered when
        for j in unpaired_dets:
            self.tracks.append(Track(self.params, detections[j], info[j], self.id))      # Create a new track object with the incoming detection and assign it a unique id
            self.id += 1      # increment ID

        return valid_detections