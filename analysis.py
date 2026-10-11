"""
Energy development profiles
---------------------------
Two questions, answered with public data for 164 countries (2022):

1. Which countries share the same energy development profile?   -> k-means clustering
2. How closely does electricity generated per person track wealth per person? -> linear regression

Data: Our World in Data, energy dataset (https://github.com/owid/energy-data),
which compiles Ember, the Energy Institute Statistical Review and the
Maddison Project Database (GDP).

Run:  python analysis.py
Needs: pandas, numpy, scikit-learn, matplotlib
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

YEAR = 2022
RAW = Path("data/owid-energy-data.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. Data
df = pd.read_csv(RAW)
df = df[df["iso_code"].notna() & ~df["iso_code"].astype(str).str.startswith("OWID")]  # countries only
df = df[df["year"] == YEAR]
cols = ["country", "iso_code", "population", "gdp", "per_capita_electricity", "low_carbon_share_elec"]
df = df[cols].dropna()
df = df[df["per_capita_electricity"] > 0]

df["gdp_per_capita"] = df["gdp"] / df["population"]
df["log_elec_pc"] = np.log10(df["per_capita_electricity"])  # kWh generated per person, log scale
df["log_gdp_pc"] = np.log10(df["gdp_per_capita"])
df.to_csv("data/countries_2022.csv", index=False)
print(f"{len(df)} countries in {YEAR}")

# ---------------------------------------------------------------- 2. K-means clustering
# Two features describe an energy profile: how much electricity people use,
# and how much of it is low-carbon (renewables + nuclear).
features = df[["log_elec_pc", "low_carbon_share_elec"]].to_numpy()
X = StandardScaler().fit_transform(features)  # put both features on the same scale

scores = {}
for k in range(2, 8):
    labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)
    scores[k] = silhouette_score(X, labels)
    print(f"k={k}  silhouette={scores[k]:.3f}")

K = 4  # highest silhouette score (see README)
km = KMeans(n_clusters=K, n_init=20, random_state=42).fit(X)
df["cluster"] = km.labels_

# Name each cluster from its average profile
profile = df.groupby("cluster").agg(
    countries=("country", "count"),
    elec_kwh=("per_capita_electricity", "median"),
    lowcarbon=("low_carbon_share_elec", "median"),
    gdp_pc=("gdp_per_capita", "median"),
)
print(profile.round(1))

def name(row):
    high = row.elec_kwh >= profile.elec_kwh.median()
    green = row.lowcarbon >= 50
    if high and green:
        return "Abundant & low-carbon"
    if high:
        return "Abundant & fossil-based"
    if green:
        return "Scarce & low-carbon"
    return "Scarce & fossil-based"

names = {c: name(r) for c, r in profile.iterrows()}
df["profile"] = df["cluster"].map(names)

# ---------------------------------------------------------------- 3. Linear regression
# log10(electricity per person) = a + b * log10(GDP per person)
reg = LinearRegression().fit(df[["log_gdp_pc"]], df["log_elec_pc"])
slope, intercept = float(reg.coef_[0]), float(reg.intercept_)
r2 = float(reg.score(df[["log_gdp_pc"]], df["log_elec_pc"]))
df["predicted_log_elec"] = reg.predict(df[["log_gdp_pc"]])
df["residual"] = df["log_elec_pc"] - df["predicted_log_elec"]  # >0: more electricity than income predicts
print(f"slope={slope:.2f}  intercept={intercept:.2f}  R2={r2:.2f}")

# ---------------------------------------------------------------- 4. Export
summary = {
    "year": YEAR,
    "countries": int(len(df)),
    "silhouette": {str(k): round(v, 3) for k, v in scores.items()},
    "k": K,
    "regression": {"slope": round(slope, 3), "intercept": round(intercept, 3), "r2": round(r2, 3)},
    "profiles": {
        names[c]: {
            "countries": int(r.countries),
            "median_kwh_per_person": round(float(r.elec_kwh)),
            "median_low_carbon_share": round(float(r.lowcarbon), 1),
            "median_gdp_per_person": round(float(r.gdp_pc)),
        }
        for c, r in profile.iterrows()
    },
    "most_above_prediction": df.nlargest(5, "residual")["country"].tolist(),
    "most_below_prediction": df.nsmallest(5, "residual")["country"].tolist(),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))

points = df[["country", "iso_code", "per_capita_electricity", "low_carbon_share_elec",
             "gdp_per_capita", "profile", "residual"]].round(3)
points.to_csv(OUT / "countries_with_profiles.csv", index=False)

# ---------------------------------------------------------------- 5. Charts
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Abundant & low-carbon": "#006039", "Abundant & fossil-based": "#0E0F0E",
              "Scarce & low-carbon": "#7FAF97", "Scarce & fossil-based": "#A37E2C"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for p, g in df.groupby("profile"):
        ax[0].scatter(g.low_carbon_share_elec, g.per_capita_electricity, s=22, c=colors[p], label=p, alpha=.85)
    ax[0].set_yscale("log"); ax[0].set_xlabel("Low-carbon share of electricity (%)")
    ax[0].set_ylabel("Electricity generated per person (kWh, log)"); ax[0].set_title(f"K-means: {K} energy profiles ({YEAR})")
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].scatter(df.gdp_per_capita, df.per_capita_electricity, s=18, c="#9AA29D")
    xs = np.linspace(df.log_gdp_pc.min(), df.log_gdp_pc.max(), 50)
    ax[1].plot(10 ** xs, 10 ** (intercept + slope * xs), c="#006039", lw=2)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("GDP per person (log)"); ax[1].set_ylabel("Electricity generated per person (kWh, log)")
    ax[1].set_title(f"Regression: slope {slope:.2f}, R² {r2:.2f}")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "charts.png", dpi=150)
except ImportError:
    print("matplotlib not installed: charts skipped")

print("Done. See results/")
