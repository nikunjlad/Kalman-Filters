import numpy as np
from scipy.optimize import linear_sum_assignment

def generate_cost_matrix(tracks, detections):

    M = len(tracks)         # The length of the predicted tracks
    N = len(detections)     # The length of the detections

    cost_matrix = np.zeros(shape=(M,N))    # Create an empty MxN matrix of zeros to hold the cost of association

    for i in range(0, M):
        for j in range(0, N):
            dist = tracks[i].predicted[:2] - detections[j].reshape((2,1))
            cost_matrix[i][j] = np.sqrt(dist[0]**2 + dist[1]**2)

    max_value = np.max(cost_matrix)     # find the maximum value in the matrix
    cost_matrix = cost_matrix / max_value   # normalize the matrix by dividing the cost_matrix by the max value to have the cost in 0-1 range

    return cost_matrix

def track_association(tracks, detections, dist_th = 0.5):

    if not tracks:
        return [], np.arange(len(detections)), []

    # Get the Cost i.e the Euclidean distance between each predicted track and measured detection
    cost_matrix = generate_cost_matrix(tracks, detections)

    # Use Hungarian Algorithm to find optimal matches
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    optimal_associations = list(zip(row_indices, col_indices))

    # Find unpaired detections and trackers
    unpaired_tracks = [d for d, _ in enumerate(tracks) if d not in row_indices]
    unpaired_detections = [t for t, _ in enumerate(detections) if t not in col_indices]

    # Filter out matches with distance greater than threshold
    pairs = []
    for i, j in optimal_associations:
        if cost_matrix[i][j] < dist_th:
            pairs.append((i,j))
        else:
            unpaired_tracks.append(i)
            unpaired_detections.append(j)

    if len(unpaired_detections) > 1:
        unpaired_detections.sort()

    if len(unpaired_tracks) > 1:
        unpaired_tracks.sort()

    return pairs, unpaired_detections, unpaired_tracks
