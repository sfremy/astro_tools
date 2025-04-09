import pandas as pd

source_fn = 'HARPS_RVBank_ver02.csv'

# Separate observations for each source and save
# to individual files.
source_table = pd.read_csv(source_fn)
target_list = set(source_table['target'])

for target in target_list:
    
    target_table = source_table[source_table['target'] == target]
    target_table.to_csv(f'{target}.csv')