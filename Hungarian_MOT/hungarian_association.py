import numpy as np
from scipy.optimize import linear_sum_assignment

cost_matrix = np.array([[44.5, 50.5, 81.0, 49.0],
                        [24.5, 48.5, 40.0, 20.0],
                        [14.5, 42.5, 20.0, 8.0],
                        [26.0, 64.0, 24.5, 14.5]])

row_indices, col_indices = linear_sum_assignment(cost_matrix)
opt_associations = list(map(lambda x, y: (x, y), row_indices, col_indices))
min_cost = cost_matrix[row_indices, col_indices].sum()

print(f"Row Indices: {row_indices}")
print(f"Column Indices: {col_indices}")
print(f"Optimal Association: {opt_associations}")
print(f"Minimum Cost: {min_cost}")