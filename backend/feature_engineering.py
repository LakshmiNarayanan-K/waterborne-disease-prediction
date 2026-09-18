import pandas as pd
import numpy as np


def add_features(df):
    """
    Add the three engineered features used by the trained models.
    """

    df = df.copy()

    # 1. Coliform ratio
    coliform_ratio = (
        df["fecal_coliform_per_100ml"] /
        (df["total_coliform_per_100ml"] + 1)
    )

    # 2. Log coliform ratio
    df["log_coliform_ratio"] = np.log1p(coliform_ratio)

    # 3. BOD / DO ratio
    df["bod_do_ratio"] = (
        df["bod_mg_l"] /
        (df["dissolved_oxygen_mg_l"] + 1e-6)
    )

    # 4. Sanitation risk
    df["sanitation_risk"] = (
        df["open_defecation_rate"] *
        (1 - df["sewage_treatment_pct"] / 100)
    )

    return df