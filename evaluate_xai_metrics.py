import json
import pandas as pd

# Load the JSON data from xai_metrics.txt
with open('xai_metrics2.txt', 'r') as f:
    data = json.loads(f.read())

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
# Set display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df)

# Create a function to get base metric name (removing MORF/LERF)
def get_base_metric(metric):
    if 'MoRF' in metric:
        return metric.replace('(MoRF)', '').strip()
    elif 'LeRF' in metric:
        return metric.replace('(LeRF)', '').strip()
    return metric

# Add base metric column
df['base_metric'] = df['xai_metric'].apply(get_base_metric)

# Create combined values while preserving individual data points
combined_data = []
for cam in df['cam'].unique():
    for base in df['base_metric'].unique():
        morf_values = df[(df['cam'] == cam) & (df['xai_metric'].str.contains('MoRF')) & 
                        (df['base_metric'] == base)]['pertubation_value']
        lerf_values = df[(df['cam'] == cam) & (df['xai_metric'].str.contains('LeRF')) & 
                        (df['base_metric'] == base)]['pertubation_value']
        
        if not morf_values.empty and not lerf_values.empty:
            combined_values = (morf_values.values + lerf_values.values) / 2
            for value in combined_values:
                combined_data.append({
                    'cam': cam,
                    'xai_metric': base + ' (Combined)',
                    'base_metric': base,
                    'pertubation_value': value
                })

# Create DataFrame with combined values
df_combined = pd.concat([df, pd.DataFrame(combined_data)], ignore_index=True)

# Calculate statistics for all metrics including combined
stats = df_combined.groupby(['cam', 'xai_metric'])['pertubation_value'].agg(['mean', 'std']).round(4)
stats.columns = ['Mean', 'Std']

print("\nStatistics for each CAM and XAI metric combination (including combined):")
print("================================================================")
print(stats)

# Optional: If you want to reset the index for better readability
stats_reset = stats.reset_index()
print("\nAlternative format:")
print(stats_reset)
