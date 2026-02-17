import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

TRAIN_NPZ = os.path.join(PROJECT_DIR, "data", "dreams_units", "train.npz")
TEST_NPZ  = os.path.join(PROJECT_DIR, "data", "dreams_units", "test.npz")
OUT_DIR   = os.path.join(PROJECT_DIR, "dataset", "DREAMS")

def write_ts(npz_path, out_ts_path, split_name):
    data = np.load(npz_path)
    X = data["X"][:, :, 0]
    y = data["y"]

    with open(out_ts_path, "w") as f:
        f.write("@problemName DREAMS\n")
        f.write("@timeStamps false\n")
        f.write("@missing false\n")
        f.write("@univariate true\n")
        f.write("@classLabel true 0 1\n")
        f.write("@data\n")

        for i in range(len(X)):
            series = ",".join([f"{v:.6f}" for v in X[i]])
            f.write(f"{series}:{int(y[i])}\n")

    print(f"[OK] {split_name}: {out_ts_path}")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    write_ts(
        TRAIN_NPZ,
        os.path.join(OUT_DIR, "DREAMS_TRAIN.ts"),
        "TRAIN"
    )

    write_ts(
        TEST_NPZ,
        os.path.join(OUT_DIR, "DREAMS_TEST.ts"),
        "TEST"
    )
