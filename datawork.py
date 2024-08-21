import pandas as pd

# from sklearn

#def read_parquet(file):
#    result = []
#    data = pd.read_parquet(file)
#    for index in data.index:
#        res = data.loc[index].values[0:-1]
#        result.append(res)
#    print(result)

#read_parquet('train-0.parquet')


data = pd.read_parquet('train-0.parquet', engine='pyarrow')
df = pd.DataFrame([[d['en'], d['me']] for d in data.translation],
columns=['English', 'Middle English'])
#print(df.loc[[55540]])
df.to_csv('train-0.csv', index=False, escapechar='$', quoting=2)
