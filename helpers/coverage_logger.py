import pandas as pd
import os
from collections import defaultdict


class CoverageLoggerCSV:
    def __init__(self, output_dir="coverage_snapshots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.counter = defaultdict(lambda: [0.0, 0])

    def log(self, x, y, direction_deg, reward):
        key = (x, y, direction_deg)  # no binning applied
        self.counter[key][0] += reward if reward is not None else 0.0
        self.counter[key][1] += 1

    def save(self, train_step):
        if not self.counter:
            return

        rows = []
        for (x, y, direction_deg), (reward_sum, count) in self.counter.items():
            rows.append(
                {
                    "x_pos": x,
                    "y_pos": y,
                    "direction": direction_deg,
                    "reward_sum": reward_sum,
                    "visit_count": count,
                    "train_step": train_step,
                }
            )

        df = pd.DataFrame(rows)
        filename = f"coverage_step_{train_step}.csv"
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)

        self.counter.clear()

    def reset(self):
        self.counter.clear()
