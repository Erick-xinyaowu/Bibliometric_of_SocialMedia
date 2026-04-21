import pandas as pd
from datetime import datetime

# Read the CSV file
input_file = 'data/search_contents_2026-04-15.csv'
output_file = 'data/search_contents_2026-04-15_formatted.csv'

# Read the CSV
df = pd.read_csv(input_file)

# Convert create_time to datetime and format as YYYY-MM-DD
# This ensures all dates are in the correct format
df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
df['create_time'] = df['create_time'].dt.strftime('%Y-%m-%d')

# Write to new CSV file
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"Successfully converted and saved to: {output_file}")
print(f"Total records: {len(df)}")
print("\nSample of converted dates:")
print(df[['aweme_id', 'title', 'create_time']].head(10))
