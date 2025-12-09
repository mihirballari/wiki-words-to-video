
import csv
import math

CENTER_X = 0.5
CENTER_Y = 0.5

with open("centers.csv", newline="") as f:
    reader = csv.DictReader(f)

    for i, row in enumerate(reader, start=1):
        x = float(row["x"])
        y = float(row["y"])

        dx = x - CENTER_X
        dy = y - CENTER_Y
        dist = math.hypot(dx, dy)  # Euclidean error

        print(f"row {i}: dx={dx:.4f}, dy={dy:.4f}, error={dist:.4f}")

