# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pandas",
#     "requests",
# ]
# ///

import pandas as pd

csv_url = 'https://drive.google.com/uc?id=1oYI_Vdo-Xmelq7FQVCweTQgs_Ii3gEL6'
df = pd.read_csv(csv_url)

print(df.head())
