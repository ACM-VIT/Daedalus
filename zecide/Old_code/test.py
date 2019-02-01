import pandas as pd
import datetime
file = pd.read_excel("zecide_data.xlsx","main_data")
date_present = datetime.datetime.now()
date_present = date_present.strftime("%Y-%m-%d")
date_present = "2019-01-09"
file = file.loc[file['date'] == date_present]

