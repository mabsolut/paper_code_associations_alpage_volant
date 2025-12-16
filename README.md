# Networks, but not composition or dominance, capture early community responses to warming in alpine grassland

This repository contains the data and code used to reproduce the analyses presented in the manuscript.

---

## 1. Installation

All analyses were conducted using Python (≥3.9).

- Create a virtual env in the root folder : `python -m venv venv`
- Activate the environment : `source venv/bin/activate`
- upgrade pip : `pip install --upgrade pip`
- Install the requierments : `pip install -r requirements.txt`

## 2. Data 

data/pinpoints.csv

Each row corresponds to the observation of a plant species at a single pinpoint.

Column:

- "subplotmaincode": Identifier of the community, grouping several spatial units
- "Species": Species name
- "comment": Optional comment 
- "pinpoint": Pinpoint number (1–25) within one of the four subcommunities
- "number_of_obs": Number of times the species touched the pinpoint
- "Year": Year of the survey
- "Site_Treatment": Experimental treatement ("G_CP": alpine, "L_CP": subalpine, "L_TP": alpine warmed)
- "Replicate": Replicate number (1–10)
- "Subplot": Subplot identifier (one for the alpine and subalpine treatments: "0"; and four for the alpine warmed treatment: "A", "B", "C", "D"), each subplot correspond to one community
- "Subplot2": Sub-subplot identifier ("LL", "LR", "UL", "UR"), four per community

## 3. Code

The script main.py allows all analyses to be run sequentially.

Analyses were performed under Linux; multiprocessing is more reliable on this operating system.

Some steps (notably the computation of all possible effect-size combinations) are computationally intensive and may take up to ~3 hours. A precomputed option is available to skip these long steps if needed.

## 4. Reproducibility

All figures and results presented in the manuscript can be reproduced using the scripts in this repository, starting from main.py.

## 5. Software versions

Analyses were conducted under:
- Python 3.12.3
- Linux (Ubuntu 24.04.2 LTS)
