import csv
import numpy as np

rows_x = []
rows_y = []

with open("football_raw.csv", mode="r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    headers = None
    for row in reader:
        if not row:
            continue
        if row[0] == "Rk":
            headers = row
            continue
        if headers is None:
            continue

        # skip blank/summary rows without a numeric Rk
        try:
            int(row[0])
        except ValueError:
            continue

        idx_90s = headers.index("90s")
        idx_x = headers.index("Sh/90")
        idx_y = headers.index("SoT/90")

        try:
            n90 = float(row[idx_90s])
            val_x = float(row[idx_x])
            val_y = float(row[idx_y])
        except (ValueError, IndexError):
            continue

        if n90 >= 3.0:
            rows_x.append(val_x)
            rows_y.append(val_y)

x_arr = np.array(rows_x, dtype=np.float64)
y_arr = np.array(rows_y, dtype=np.float64)

rng = np.random.default_rng(42)
if len(x_arr) > 120:
    chosen = rng.choice(len(x_arr), size=120, replace=False)
    x_arr = x_arr[chosen]
    y_arr = y_arr[chosen]

x_norm = (x_arr - np.mean(x_arr)) / np.std(x_arr)
y_norm = (y_arr - np.mean(y_arr)) / np.std(y_arr)

with open("football_data.csv", mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y"])
    for xv, yv in zip(x_norm, y_norm):
        writer.writerow([f"{xv:.18e}", f"{yv:.18e}"])

print(f"Saved {len(x_norm)} rows to football_data.csv")
for i in range(min(8, len(x_norm))):
    print(f"{x_norm[i]:.18e},{y_norm[i]:.18e}")