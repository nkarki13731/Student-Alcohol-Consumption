# features.py
import numpy as np

def add_engineered_features(df):
    
    out = df.copy()
    out["goout_degree"]   = np.where(out["goout"] > 3, "high", "low")
    out["freetime_degree"]= np.where(out["freetime"] > 3, "high", "low")
    out["health_degree"]  = np.where(out["health"] > 3, "high", "low")
    out["famrel_degree"]  = np.where(out["famrel"] > 3, "high", "low")
    out["alc_level"]      = np.where(((out["Dalc"] + out["Walc"])/2) > 3, "High", "Low")
    return out
