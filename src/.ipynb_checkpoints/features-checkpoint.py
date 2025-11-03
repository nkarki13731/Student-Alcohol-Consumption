mergeddf["goout_degree"] = np.where(mergeddf["goout"] > 3, "high", "low")
mergeddf["alc_level"] = np.where(((mergeddf["Dalc"]+mergeddf["Walc"])/2)> 3,"High","Low")
mergeddf["freetime_degree"] = np.where(mergeddf["freetime"] > 3, "high", "low")
mergeddf["health_degree"] = np.where(mergeddf["health"] > 3, "high", "low")
mergeddf["famrel_degree"] = np.where(mergeddf["famrel"] > 3, "high", "low")
featureset= ['famsup', 'goout_degree',]
g2 = mergeddf['G2'].copy()
Q1 = g2.quantile(0.25)
Q3 = g2.quantile(0.75)
IQR = Q3 - Q1
lo = Q1 - 1.5*IQR
hi = Q3 + 1.5*IQR

mergeddf_win = mergeddf.copy()
mergeddf_win['G2'] = mergeddf_win['G2'].clip(lower=lo, upper=hi)