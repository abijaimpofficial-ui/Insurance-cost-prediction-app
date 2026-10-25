"""
generate_dataset.py
-------------------
Creates data/insurance.csv with the SAME columns as the well-known Kaggle
"Medical Cost Personal Datasets" (age, sex, bmi, children, smoker, region, charges).

NOTE: This is a synthetic stand-in (1338 rows, same structure and similar
relationships). If you have internet access, download the real file from
https://www.kaggle.com/datasets/mirichoi0218/insurance and replace
data/insurance.csv with it - the main script works unchanged.
"""
import numpy as np
import pandas as pd
import os

def generate(n=1338, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 65, n)
    sex = rng.choice(["male", "female"], n)
    bmi = np.clip(rng.normal(30.6, 6.1, n), 15.9, 53.1).round(2)
    children = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.43, 0.24, 0.18, 0.12, 0.02, 0.01])
    smoker = rng.choice(["yes", "no"], n, p=[0.205, 0.795])
    region = rng.choice(["southwest", "southeast", "northwest", "northeast"], n)

    s = (smoker == "yes").astype(int)
    charges = (
        -3000
        + 260 * age
        + 330 * children
        + 24000 * s
        + 440 * (bmi - 30) * s          # BMI matters much more for smokers
        + 90 * np.maximum(bmi - 30, 0)
        + np.where(region == "southeast", 600, 0)
        + np.where(sex == "male", 150, 0)
        + rng.gamma(2.0, 1500, n) + rng.normal(0, 2500, n)   # noise
    )
    charges = np.clip(charges, 1121.87, 63770.43).round(2)
    return pd.DataFrame(dict(age=age, sex=sex, bmi=bmi, children=children,
                             smoker=smoker, region=region, charges=charges))

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    generate().to_csv("data/insurance.csv", index=False)
    print("Saved data/insurance.csv")
