import pandas as pd



data = pd.read_excel('ME_cleaned.ods', engine='odf')
print(data)
#print(df.loc[[55540]])
columns_titles = ["Middle English", "English"]
data = data.reindex(columns=columns_titles)
data.to_csv('ME_cleaned.csv', index=False, escapechar='$', quoting=2)
