#!/usr/bin/env python3
"""
Script to analyze the structure and meaning of generated CSV metric files.
"""
import pandas as pd
import os

def analyze_csv_structure(filepath):
    """Analyze the structure of a CSV file."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    print(f"\n=== ANALYZING: {os.path.basename(filepath)} ===")
    df = pd.read_csv(filepath)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    if 'metric' in df.columns:
        print(f"\nUnique metrics and their frequencies:")
        metric_counts = df['metric'].value_counts()
        for metric, count in metric_counts.items():
            print(f"  {metric}: {count} records")
    
    if 'traffic_light_id' in df.columns:
        print(f"\nUnique traffic lights:")
        tl_counts = df['traffic_light_id'].value_counts()
        for tl, count in tl_counts.items():
            print(f"  {tl}: {count} records")
    
    if 'time_step' in df.columns:
        print(f"\nTime step range: {df['time_step'].min()} to {df['time_step'].max()}")
        print(f"Unique time steps: {df['time_step'].nunique()}")
    
    if 'traffic_light_id' in df.columns and 'metric' in df.columns:
        print(f"\nRecords per traffic light and metric:")
        structure = df.groupby(['traffic_light_id', 'metric']).size()
        for (tl, metric), count in structure.items():
            print(f"  {tl} - {metric}: {count} records")

def main():
    """Main analysis function."""
    base_path = "models/model_10"
    
    # Find all CSV files in the model directory
    csv_files = []
    if os.path.exists(base_path):
        for file in os.listdir(base_path):
            if file.endswith('.csv'):
                csv_files.append(os.path.join(base_path, file))
    
    if not csv_files:
        print("No CSV files found in models/model_10/")
        return
    
    for csv_file in sorted(csv_files):
        analyze_csv_structure(csv_file)
    
    print("\n" + "="*80)
    print("DATA STRUCTURE EXPLANATION:")
    print("="*80)
    print("""
    The CSV files contain metrics collected during traffic light simulations.
    Each row represents one metric measurement at one time step for one traffic light.
    
    Column meanings:
    - traffic_light_id: The ID of the traffic light intersection
    - metric: The type of measurement (e.g., agent_reward, system_density, travel_speed)
    - time_step: The simulation time step when this measurement was taken
    - value: The actual measured value for this metric
    - episode: The training episode number
    - simulation_type: The type of simulation (dqn_huber, dqn_weighted, qlearning, baseline)
    
    Why data lengths vary:
    1. Different metrics are collected at different frequencies
    2. Some metrics may only be available when certain conditions are met
    3. Agent-specific metrics (like rewards) are only collected during agent actions
    4. System-wide metrics (like density) may be collected every step
    5. Some metrics may be missing if the traffic light had no activity
    
    Each episode typically runs for 3600 simulation steps (60 steps × 60 data points).
    The number of records per metric depends on how often that metric is updated.
    """)

if __name__ == "__main__":
    main()
