import pandas as pd
import numpy as np

# Concatenating all tracking weeks
tracking_weeks = []
for week_num in range(1, 10):
    file_path = f'data/tracking_week_{week_num}.csv'
    week_data = pd.read_csv(file_path)
    tracking_weeks.append(week_data)

tracking_weeks = pd.concat(tracking_weeks)

# Merge play and tackle data
plays = pd.read_csv('data/plays.csv')
tackles = pd.read_csv('data/tackles.csv')
plays_plus_tackles = pd.merge(plays, tackles, on=['gameId', 'playId'], how='inner')

# Reading necessary data
short_tracking_weeks = tracking_weeks[['nflId', 's', 'a']]
games = pd.read_csv('data/games.csv')
players = pd.read_csv('data/players.csv')

# Filtering defensive players
defensive_players = players[players['position'].isin(['CB', 'DB', 'DE', 'DT', 'FS', 'ILB', 'LS', 'MLB', 'NT', 'OLB', 'SS'])]

# Extracting max 'a' and 's' values for each defensive player
defensiveids = defensive_players['nflId'].unique()
max_a_values, max_s_values = [], []

for player in defensiveids:
    player_data = short_tracking_weeks[short_tracking_weeks['nflId'] == player]
    max_a = player_data['a'].max()
    max_s = player_data['s'].max()
    max_a_values.append(max_a)
    max_s_values.append(max_s)

defensive_players['max_a'] = max_a_values
defensive_players['max_s'] = max_s_values

# Aggregating tackle-related statistics
player_tackle_df = plays_plus_tackles.groupby("nflId").agg(
    tackles=pd.NamedAgg(column='tackle', aggfunc='sum'),
    assists=pd.NamedAgg(column='assist', aggfunc='sum'),
    forced_fumbles=pd.NamedAgg(column='forcedFumble', aggfunc='sum'),
    missed_tackles=pd.NamedAgg(column='pff_missedTackle', aggfunc='sum'),
).reset_index()

# Calculating tackle efficiency and additional metrics
player_tackle_df['total_tackles'] = player_tackle_df['tackles'] + (player_tackle_df['assists'] * 0.5)
player_tackle_df['tackle_efficiency'] = player_tackle_df['total_tackles'] / (
        player_tackle_df['total_tackles'] + player_tackle_df['missed_tackles'])

# Merging tackle data with player information
player_tackles = pd.merge(defensive_players, player_tackle_df, how="inner", on=['nflId'])

# Grouping by position to calculate average tackles
position_tackles_df = player_tackles.groupby("position").agg(
    avg_tackles_by_pos=pd.NamedAgg(column='total_tackles', aggfunc='mean')
).reset_index()

# Function to set average tackles by position
def set_average_tackles_by_pos(position):
    row = position_tackles_df['position'] == position
    column = 'avg_tackles_by_pos'
    avg_tackles = position_tackles_df.loc[row, column].iloc[0]
    return avg_tackles

# Function to calculate BMI
def calculate_bmi(row):
    height = row['height']
    weight = row['weight']
    feet, inches = height.split('-')
    total_height_inches = int(feet) * 12 + int(inches)
    bmi = (weight / (total_height_inches ** 2)) * 703
    return bmi

# Applying functions to create new columns
player_tackles['avg_tackles_by_pos'] = player_tackles['position'].apply(set_average_tackles_by_pos)
player_tackles['bmi'] = player_tackles.apply(calculate_bmi, axis=1)
player_tackles['tackle_factor'] = player_tackles['total_tackles'] / player_tackles['avg_tackles_by_pos']

# Calculating and adding the 75th percentile of tackle factor for each position
percentile_75 = player_tackles.groupby('position')['tackle_factor'].quantile(0.75).reset_index()
percentile_75.columns = ['position', '75th_percentile_tackle_factor']

# Merging percentile values into the player_tackles DataFrame based on position
player_tackles = pd.merge(player_tackles, percentile_75, how='left', on='position')

# Categorizing players based on tackle factor percentiles
player_tackles['75th_percentile_category'] = np.where(
    player_tackles['tackle_factor'] > player_tackles['75th_percentile_tackle_factor'],
    'Above',
    'Below'
)

# Dropping unnecessary columns
player_tackles = player_tackles.drop(columns=[
    'avg_tackles_by_pos', 'height', 'birthDate', 'collegeName',
    'tackles', 'assists', 'forced_fumbles', 'missed_tackles',
    'total_tackles', 'weight']
)

# Saving processed data to a CSV file and printing the DataFrame
player_tackles.to_csv('data/player_tackles_ML.csv', index=False)
print(player_tackles)