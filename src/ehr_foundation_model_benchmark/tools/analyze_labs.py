
# how to create measurement_unit_counts.csv
# before processing

from tqdm import tqdm
import pandas as pd
import gc
import pymssql

server = 'your_server_name_or_ip'
user = 'your_username'
password = 'your_password'
database = 'your_database'

conn = pymssql.connect(server=server, user=user, password=password, database=database)
print("Connection successful!")

# query to get counts per unit per measurement
labs_query = """SELECT DISTINCT measurement_concept_id, c1.concept_name as measurement_name, unit_concept_id, c2.concept_name as unit_concept_name, COUNT(measurement_id)
FROM dbo.measurement
LEFT JOIN dbo.concept as c1 on c1.concept_id = measurement_concept_id
LEFT JOIN dbo.concept as c2 on c2.concept_id = unit_concept_id
GROUP BY measurement_concept_id, c1.concept_name, unit_concept_id, c2.concept_name"""

list_chunks = []
for chunk in tqdm(pd.io.sql.read_sql(labs_query, conn, chunksize=10000000)):
    list_chunks.append(chunk)

df_labs = pd.concat(list_chunks)

del list_chunks
gc.collect()

df_labs.rename({'':'counts'}, axis=1, inplace=True)

df_labs.to_csv('measurement_unit_counts.csv')




