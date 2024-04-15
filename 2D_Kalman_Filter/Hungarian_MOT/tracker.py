import numpy as np
from kalman_filter_2d import KalmanFilter2D
from tracker_utils import track_association

class Track():

    def __init__(self, params, detection, trackId):
        self.track_id = trackId     # unique track ID for each detected object
        self.KF = KalmanFilter2D(params, detection)     # a Kalman Filter for each Object to be detected in space
        self.num_lost_dets = 0          # to keep track of number of times the variable was lost in the frame
        self.trace = []                 # to keep track of the trace route of the object in space. A trace route is the last x points it travelled in space

    def update(self, detection):        # get the updated positions of the KF using the new measured detections
        self.updated = self.KF.update(detection)

    def predict(self):                  # predict the next state of the object's position using KF
        self.predicted = self.KF.predict()

class Tracker():

    def __init__(self, params):
        self.params = params        # read the parameters from the JSON file
        self.min_dist = self.params["min_dist"]     # get the minimum distance the centers should have to be considered a valid candidate for association
        self.max_lost_dets = self.params["max_lost_dets"]
        self.trace_length = self.params["trace_length"]
        self.id = self.params["id"]
        self.tracks = []

    def manage_tracks(self, detections):

        # given new detections (or measurements), first predict the next state of the objects in space using the 2D Kalman Filter predict function
        for i in range(len(self.tracks)):
            self.tracks[i].predict()        # Update each track in track list with new predicted states

        # given the new predicted states of the object, we do track association.
        # Basically we match currently tracked objects (with the predicted states) to the new measurements
        # We create pairs between which objects match best from previous time frame to current frame. Likewise, we keep track of unpaired detections
        # and tracks.
        pairs, unpaired_dets, unpaired_tracks = track_association(self.tracks, detections, self.min_dist)

        # For Associated pairs
        # For objects whose associations were successful between predicted and measured positions in space, update the trace list
        for i,j in pairs:
            self.tracks[i].num_lost_dets = 0        # Reset the ith track lost detections counter to 0. That's because the object is still present hence not lost

            # Update the ith track after association with the measurement information.
            # The updation step corrects any errors made in the Kalman Filter prediction using the incoming detection measurements
            self.tracks[i].update(detections[j])

            # if length of trace is greater than 40 (since we are keeping a history of past 40 points the object traversed in space), remove the oldest one
            if len(self.tracks[i].trace) >= self.trace_length:
                self.tracks[i].trace = self.tracks[i].trace[:-1]    # remove oldest track

            # Insert new updated position of the object in the trace list (which the KF updated earlier).
            self.tracks[i].trace.insert(0, self.tracks[i].updated)

        # FOR UNPAIRED TRACKS
        # check and delete each track that has undergone lost detection for more than max number of frames
        del_track = 0
        for i in unpaired_tracks:
            if self.tracks[i - del_track].num_lost_dets > self.max_lost_dets:    # if track lost more than threshold, delete it
                print(unpaired_tracks, len(self.tracks))
                del self.tracks[i - del_track]      # delete tracked object if lost for a long time
                del_track += 1
            else:
                self.tracks[i - del_track].num_lost_dets += 1       # if object was just missed once, then increment counter

        # FOR UNPAIRED DETECTIONS
        # This loop is triggered when new detections come and they are not yet assigned a track
        # This loop is also triggered when
        for j in unpaired_dets:
            self.tracks.append(Track(self.params, detections[j], self.id))
            self.id += 1



