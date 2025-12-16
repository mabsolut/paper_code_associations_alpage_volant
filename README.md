# Networks, but not composition or dominance, capture early community responses to warming in alpine grassland

## Fisrt install

- Create a virtual env in the root folder : `python -m venv venv`
- Activate the environment : `source venv/bin/activate`
- upgrade pip : `pip install --upgrade pip`
- Install the requierments : `pip install -r requirements.txt`

## Data 

data/pinpoints.csv
Each row represent the observetion of a species at a pinpoint
Column:
- "subplotmaincode": code of the community regouping different following informations
- "Species": species name
- "comment": comment if nedded 
- "pinpoint": number of the pinpoint (from 1 to 25) within on of the 4 subcommunities 
- "number_of_obs": number of time that this species touche the pinpoint
- "Year": year of survey
- "Site_Treatment": treatement ("G_CP": alpine, "L_CP": subalpine, "L_TP": alpine warmed)
- "Replicate": number of the replicate from 1 to 10
- "Subplot": name of the subplot (there is only one subplot for the alpine and subalpine treatment: "0", and 4 subplots for the alpine warmed treatment: "A", "B", "C", "D"), each subplot is a community
- "Subplot2": name of the subsubplot (there are 4 by community: "LL", "LR", "UL", "UR")
