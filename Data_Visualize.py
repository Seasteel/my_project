import pandas as pd
file = 'owid-covid-data.xlsx'
df = pd.read_excel(file)
df = df.sort_values(by='total_cases',ascending=False)
df['location'].unique()
world = df[df['location'] == 'World']
mgl = df[df['location'] == 'Mongolia']
world.sort_values(by='new_cases', ascending=False)
world = world.pivot_table(index='date')
world['new_cases'].plot()