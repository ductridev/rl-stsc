import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional

class SimulationComparison:
    """Utility class for comparing metrics across different simulation types."""
    
    def __init__(self, path: str):
        self.path = path
        
    def combine_simulation_dataframes(self, episode: int, simulation_types: List[str] = None) -> pd.DataFrame:
        """
        Combine DataFrames from all simulation types for a specific episode.
        
        Args:
            episode (int): Episode number to compare
            simulation_types (List[str]): List of simulation types to include
            
        Returns:
            pd.DataFrame: Combined DataFrame with all simulation data
        """
        if simulation_types is None:
            simulation_types = ['baseline', 'q_learning', 'dqn_mse', 'dqn_huber', 'dqn_weighted', 'dqn_qr']
        
        combined_dfs = []
        
        for sim_type in simulation_types:
            filename = f"{self.path}{sim_type}_metrics_episode_{episode}.csv"
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                combined_dfs.append(df)
                print(f"Loaded {sim_type} data: {len(df)} records")
            else:
                print(f"Warning: File not found: {filename}")
        
        if combined_dfs:
            combined_df = pd.concat(combined_dfs, ignore_index=True)
            return combined_df
        else:
            print("No data files found for episode", episode)
            return pd.DataFrame()
    
    def create_traffic_light_comparison_table(self, episode: int, metrics: List[str] = None, simulation_types: List[str] = None) -> pd.DataFrame:
        """
        Create a comparison table showing metrics for each traffic light across all simulations.
        
        Args:
            episode (int): Episode number to compare
            metrics (List[str]): List of metrics to include in comparison
            simulation_types (List[str]): List of simulation types to include
            
        Returns:
            pd.DataFrame: Comparison table with traffic lights as rows and simulation types as columns
        """
        if metrics is None:
            metrics = ['density', 'travel_speed', 'travel_time', 'outflow', 'queue_length', 'waiting_time']
        
        combined_df = self.combine_simulation_dataframes(episode, simulation_types)
        
        if combined_df.empty:
            return pd.DataFrame()
        
        # Filter for specified metrics
        filtered_df = combined_df[combined_df['metric'].isin(metrics)]
        
        # Calculate average values per traffic light per simulation type per metric
        avg_df = filtered_df.groupby(['traffic_light_id', 'simulation_type', 'metric'])['value'].mean().reset_index()
        
        comparison_tables = {}
        
        for metric in metrics:
            metric_df = avg_df[avg_df['metric'] == metric]
            
            # Pivot to get simulation types as columns
            pivot_df = metric_df.pivot(index='traffic_light_id', columns='simulation_type', values='value')
            pivot_df.name = metric
            comparison_tables[metric] = pivot_df
        
        return comparison_tables
    
    def create_summary_comparison_table(self, episode: int, metrics: List[str] = None, simulation_types: List[str] = None) -> pd.DataFrame:
        """
        Create a summary table showing overall performance across all traffic lights.
        
        Args:
            episode (int): Episode number to compare
            metrics (List[str]): List of metrics to include
            simulation_types (List[str]): List of simulation types to include
            
        Returns:
            pd.DataFrame: Summary table with metrics as rows and simulation types as columns
        """
        if metrics is None:
            metrics = ['density', 'travel_speed', 'travel_time', 'outflow', 'queue_length', 'waiting_time']
        
        combined_df = self.combine_simulation_dataframes(episode, simulation_types)
        
        if combined_df.empty:
            return pd.DataFrame()
        
        # Filter for specified metrics
        filtered_df = combined_df[combined_df['metric'].isin(metrics)]
        
        # Calculate overall average across all traffic lights
        summary_df = filtered_df.groupby(['simulation_type', 'metric'])['value'].mean().reset_index()
        
        # Pivot to get simulation types as columns
        pivot_df = summary_df.pivot(index='metric', columns='simulation_type', values='value')
        
        return pivot_df
    
    def save_comparison_tables(self, episode: int, metrics: List[str] = None, simulation_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Save comparison tables to CSV files and return them as dictionary.
        
        Args:
            episode (int): Episode number to compare
            metrics (List[str]): List of metrics to include
            simulation_types (List[str]): List of simulation types to include
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing all comparison tables
        """
        if metrics is None:
            metrics = ['density', 'travel_speed', 'travel_time', 'outflow', 'queue_length', 'waiting_time']
        
        results = {}
        
        # Per-traffic-light comparison
        tl_comparison_tables = self.create_traffic_light_comparison_table(episode, metrics, simulation_types)
        
        if tl_comparison_tables:
            for metric, table in tl_comparison_tables.items():
                filename = f"{self.path}comparison_per_tl_{metric}_episode_{episode}.csv"
                table.to_csv(filename)
                print(f"Per-traffic-light {metric} comparison saved to: {filename}")
                results[f"per_tl_{metric}"] = table
        
        # Summary comparison
        summary_table = self.create_summary_comparison_table(episode, metrics, simulation_types)
        
        if not summary_table.empty:
            filename = f"{self.path}comparison_summary_episode_{episode}.csv"
            summary_table.to_csv(filename)
            print(f"Summary comparison saved to: {filename}")
            results["summary"] = summary_table
        
        return results
    
    def create_performance_ranking(self, episode: int, metrics: List[str] = None, simulation_types: List[str] = None) -> pd.DataFrame:
        """
        Create a performance ranking table for each metric.
        
        Args:
            episode (int): Episode number to compare
            metrics (List[str]): List of metrics to rank
            simulation_types (List[str]): List of simulation types to include
            
        Returns:
            pd.DataFrame: Ranking table showing best to worst performers
        """
        if metrics is None:
            metrics = ['density', 'travel_speed', 'travel_time', 'outflow', 'queue_length', 'waiting_time']
        
        summary_table = self.create_summary_comparison_table(episode, metrics, simulation_types)
        
        if summary_table.empty:
            return pd.DataFrame()
        
        # Define which metrics are better when higher vs lower
        higher_is_better = ['travel_speed', 'outflow']
        lower_is_better = ['density', 'travel_time', 'queue_length', 'waiting_time']
        
        ranking_data = []
        
        for metric in metrics:
            if metric in summary_table.index:
                metric_values = summary_table.loc[metric].dropna()
                
                if metric in higher_is_better:
                    # Sort descending (higher is better)
                    sorted_values = metric_values.sort_values(ascending=False)
                else:
                    # Sort ascending (lower is better)
                    sorted_values = metric_values.sort_values(ascending=True)
                
                for rank, (sim_type, value) in enumerate(sorted_values.items(), 1):
                    ranking_data.append({
                        'metric': metric,
                        'rank': rank,
                        'simulation_type': sim_type,
                        'value': value
                    })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Save ranking table
        filename = f"{self.path}performance_ranking_episode_{episode}.csv"
        ranking_df.to_csv(filename, index=False)
        print(f"Performance ranking saved to: {filename}")
        
        return ranking_df

    def print_comparison_summary(self, episode: int, metrics: List[str] = None, simulation_types: List[str] = None):
        """
        Print a formatted summary of the comparison results.
        
        Args:
            episode (int): Episode number to summarize
            simulation_types (List[str]): List of simulation types to include
        """
        print(f"\n{'='*60}")
        print(f"SIMULATION COMPARISON SUMMARY - EPISODE {episode}")
        print(f"{'='*60}")
        
        summary_table = self.create_summary_comparison_table(episode, metrics, simulation_types=simulation_types)
        ranking_df = self.create_performance_ranking(episode, metrics, simulation_types=simulation_types)
        
        if not summary_table.empty:
            print("\nOVERALL PERFORMANCE SUMMARY:")
            print("-" * 40)
            print(summary_table.round(3))
            
        if not ranking_df.empty:
            print("\nPERFORMANCE RANKINGS:")
            print("-" * 40)
            
            for metric in ranking_df['metric'].unique():
                metric_ranking = ranking_df[ranking_df['metric'] == metric]
                print(f"\n{metric.upper()}:")
                for _, row in metric_ranking.iterrows():
                    print(f"  {row['rank']}. {row['simulation_type']}: {row['value']:.3f}")
        
        print(f"\n{'='*60}")
