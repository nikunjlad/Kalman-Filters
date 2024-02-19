import numpy as np

def generate_cost_matrix(tracks, detections):

    M = len(tracks)         # The length of the predicted tracks
    N = len(detections)     # The length of the detections

    cost_matrix = np.zeros(shape=(M,N))    # Create an empty MxN matrix of zeros to hold the cost of association

    for i in range(0, M):
        for j in range(0, N):
            dx = tracks[i][0] - detections[j][0]
            dy = tracks[i][1] - detections[j][1]
            cost_matrix[i][j] = np.sqrt(dx**2 + dy**2)

    max_value = np.max(cost_matrix)     # find the maximum value in the matrix
    cost_matrix = np.round(cost_matrix / max_value, 3)   # normalize the matrix by dividing the cost_matrix by the max value to have the cost in 0-1 range

    return cost_matrix

# Define the list of predicted tracks and detections
T = [(9,9),(7,6),(5,2),(7,10)]
D = [(9,5),(7,3),(8,8),(9,11)]

# Execute cost matrix function
cost_matrix = generate_cost_matrix(T, D)
print(f"Cost Matrix: \n {cost_matrix}")
