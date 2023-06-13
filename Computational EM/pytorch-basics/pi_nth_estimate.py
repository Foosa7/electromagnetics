import torch

def estimate_pi(n):
    points_in_circle = 0
    total_points = 0

    for _ in range(n):
        x = torch.rand(1)
        y = torch.rand(1)

        distance = x ** 2 + y ** 2

        if distance <= 1:
            points_in_circle += 1
        total_points += 1

    pi = 4 * (points_in_circle / total_points)
    return pi

# Set the number of iterations for estimation
n = 1_000_000

# Estimate pi
pi_est = estimate_pi(n)
print("Estimated pi:", pi_est)
