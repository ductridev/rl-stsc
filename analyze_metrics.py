#!/usr/bin/env python3
"""
Traffic Simulation Metrics Analysis Script

This script analyzes and compares metrics from three different traffic control methods:
1. Baseline (static control)
2. Actuated (queue-based control) 
3. SKRL DQN (reinforcement learning control)

Metrics analyzed:
- Travel delay
- Travel time
- Queue length
- Waiting time
- Outflow
- Reward (for DQN only)
- Density
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class TrafficMetricsAnalyzer:
    def __init__(self, base_path: str = None):
        """Initialize the analyzer with base path for CSV files."""
        self.base_path = Path(base_path) if base_path else Path(".")
        self.data = {}
        self.metrics = ['travel_delay', 'travel_time', 'queue_length', 'waiting_time', 'outflow', 'density']
        self.simulation_types = ['baseline', 'actuated', 'skrl_dqn']
        
    def load_data(self, baseline_file: str = None, actuated_file: str = None, skrl_file: str = None):
        """Load the three CSV files."""
        files = {
            'baseline': baseline_file or 'baseline_metrics_episode_1.csv',
            'actuated': actuated_file or 'actuated_metrics_episode_1.csv', 
            'skrl_dqn': skrl_file or 'skrl_dqn_metrics_episode_1.csv'
        }
        
        for sim_type, filename in files.items():
            file_path = self.base_path / filename
            if file_path.exists():
                self.data[sim_type] = pd.read_csv(file_path)
                print(f"✅ Loaded {sim_type}: {len(self.data[sim_type])} rows")
            else:
                print(f"❌ File not found: {file_path}")
                
        if not self.data:
            raise FileNotFoundError("No CSV files could be loaded!")
            
    def get_common_metrics(self) -> List[str]:
        """Get metrics that are available in all loaded simulation types."""
        if not self.data:
            return []
        
        # Get metrics available in each simulation type
        metrics_per_sim = {}
        for sim_type, df in self.data.items():
            metrics_per_sim[sim_type] = set(df['metric'].unique())
        
        # Find intersection (common metrics)
        common_metrics = set.intersection(*metrics_per_sim.values()) if metrics_per_sim else set()
        
        # Filter out 'reward' as it's DQN-specific
        common_metrics.discard('reward')
        
        return sorted(list(common_metrics))
    
    def get_all_available_metrics(self) -> List[str]:
        """Get all unique metrics available across all simulation types."""
        all_metrics = set()
        for df in self.data.values():
            all_metrics.update(df['metric'].unique())
        return sorted(list(all_metrics))

    def get_metric_summary(self) -> pd.DataFrame:
        """Generate summary statistics for each metric and simulation type."""
        summary_data = []
        all_metrics = self.get_all_available_metrics()
        
        for sim_type, df in self.data.items():
            for metric in all_metrics:
                metric_data = df[df['metric'] == metric]['value']
                if len(metric_data) > 0:
                    summary_data.append({
                        'simulation_type': sim_type,
                        'metric': metric,
                        'sum': metric_data.sum(),
                        'mean': metric_data.mean(),
                        'median': metric_data.median(),
                        'std': metric_data.std(),
                        'min': metric_data.min(),
                        'max': metric_data.max(),
                        'count': len(metric_data)
                    })
        
        return pd.DataFrame(summary_data)
    
    def get_summary_table(self) -> pd.DataFrame:
        """Create a comprehensive summary table with sum, avg, min, max for each metric and simulation type."""
        summary = self.get_metric_summary()
        all_metrics = self.get_all_available_metrics()
        
        # Pivot the data to have simulation types as columns
        pivot_data = []
        
        for metric in all_metrics:
            metric_data = summary[summary['metric'] == metric]
            if len(metric_data) > 0:
                row_data = {'metric': metric}
                
                for _, sim_row in metric_data.iterrows():
                    sim_type = sim_row['simulation_type']
                    row_data[f'{sim_type}_sum'] = sim_row['sum']
                    row_data[f'{sim_type}_avg'] = sim_row['mean']
                    row_data[f'{sim_type}_min'] = sim_row['min']
                    row_data[f'{sim_type}_max'] = sim_row['max']
                
                pivot_data.append(row_data)
        
        return pd.DataFrame(pivot_data)
    
    def save_summary_csv(self, save_path: str = None):
        """Save summary statistics to CSV file."""
        if save_path is None:
            save_path = self.base_path / "metrics_summary.csv"
        
        # Save detailed summary
        summary = self.get_metric_summary()
        summary.to_csv(save_path, index=False)
        print(f"📊 Detailed summary saved to: {save_path}")
        
        # Save pivot table summary
        pivot_path = str(save_path).replace('.csv', '_pivot.csv')
        summary_table = self.get_summary_table()
        summary_table.to_csv(pivot_path, index=False)
        print(f"📋 Pivot summary saved to: {pivot_path}")
        
        return summary, summary_table

    def compare_performance(self) -> Dict:
        """Compare performance across simulation types for common metrics only."""
        comparison = {}
        summary = self.get_metric_summary()
        common_metrics = self.get_common_metrics()
        
        print(f"📊 Comparing performance for common metrics: {', '.join(common_metrics)}")
        if len(common_metrics) == 0:
            print("⚠️  No common metrics found across all simulation types!")
            return comparison
        
        # For each common metric, find the best performing simulation type
        for metric in common_metrics:
            metric_data = summary[summary['metric'] == metric].copy()
            if len(metric_data) == 0:
                continue
                
            # Determine if lower is better (for most metrics except outflow)
            lower_is_better = metric not in ['outflow', 'density']
            
            if lower_is_better:
                best_idx = metric_data['mean'].idxmin()
                worst_idx = metric_data['mean'].idxmax()
            else:
                best_idx = metric_data['mean'].idxmax()
                worst_idx = metric_data['mean'].idxmin()
            
            best_sim = metric_data.loc[best_idx]
            worst_sim = metric_data.loc[worst_idx]
            
            # Calculate improvement percentage
            if lower_is_better:
                improvement = ((worst_sim['mean'] - best_sim['mean']) / worst_sim['mean']) * 100
            else:
                improvement = ((best_sim['mean'] - worst_sim['mean']) / worst_sim['mean']) * 100
            
            comparison[metric] = {
                'best_simulation': best_sim['simulation_type'],
                'best_value': best_sim['mean'],
                'worst_simulation': worst_sim['simulation_type'],
                'worst_value': worst_sim['mean'],
                'improvement_pct': improvement,
                'lower_is_better': lower_is_better
            }
        
        return comparison
    
    def analyze_time_series(self) -> Dict:
        """Analyze how metrics change over time."""
        time_analysis = {}
        all_metrics = self.get_all_available_metrics()
        
        for metric in all_metrics:
            time_analysis[metric] = {}
            
            for sim_type, df in self.data.items():
                metric_data = df[df['metric'] == metric].copy()
                if len(metric_data) > 0:
                    # Sort by time_step and calculate trends
                    metric_data = metric_data.sort_values('time_step')
                    
                    # Calculate correlation with time (trend)
                    correlation = np.corrcoef(metric_data['time_step'], metric_data['value'])[0, 1]
                    
                    time_analysis[metric][sim_type] = {
                        'trend_correlation': correlation,
                        'start_value': metric_data['value'].iloc[0],
                        'end_value': metric_data['value'].iloc[-1],
                        'change_pct': ((metric_data['value'].iloc[-1] - metric_data['value'].iloc[0]) / 
                                     metric_data['value'].iloc[0] * 100) if metric_data['value'].iloc[0] != 0 else 0
                    }
        
        return time_analysis
    
    def calculate_reward_analysis(self) -> Dict:
        """Analyze reward data (only available for SKRL DQN)."""
        reward_analysis = {}
        
        if 'skrl_dqn' in self.data:
            df = self.data['skrl_dqn']
            reward_data = df[df['metric'] == 'reward']['value']
            
            if len(reward_data) > 0:
                reward_analysis = {
                    'total_reward': reward_data.sum(),
                    'average_reward': reward_data.mean(),
                    'reward_trend': np.corrcoef(range(len(reward_data)), reward_data)[0, 1],
                    'min_reward': reward_data.min(),
                    'max_reward': reward_data.max(),
                    'reward_stability': reward_data.std()
                }
        
        return reward_analysis
    
    def generate_plots(self, save_dir: str = None):
        """Generate visualization plots."""
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
        else:
            save_path = self.base_path
        
        plt.style.use('default')
        
        # Get common metrics for comparison plots and all metrics for individual plots
        common_metrics = self.get_common_metrics()
        all_metrics = self.get_all_available_metrics()
        
        # 1. Summary comparison plot (only common metrics)
        if common_metrics:
            n_metrics = len(common_metrics)
            n_cols = 3
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            fig.suptitle('Traffic Control Methods Comparison (Common Metrics)', fontsize=16, fontweight='bold')
            
            summary = self.get_metric_summary()
            
            for i, metric in enumerate(common_metrics):
                row, col = i // n_cols, i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                metric_data = summary[summary['metric'] == metric]
                if len(metric_data) > 0:
                    bars = ax.bar(metric_data['simulation_type'], metric_data['mean'], 
                                 yerr=metric_data['std'], capsize=5,
                                 color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                    ax.set_title(f'{metric.replace("_", " ").title()}')
                    ax.set_ylabel('Average Value')
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}', ha='center', va='bottom')
            
            # Hide unused subplots
            for i in range(len(common_metrics), n_rows * n_cols):
                row, col = i // n_cols, i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.set_visible(False)
            
            plt.tight_layout()
            plt.savefig(save_path / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
            print(f"📊 Saved comparison plot: {save_path / 'metrics_comparison.png'}")
        
        # 2. Time series plots (all available metrics)
        if all_metrics:
            n_metrics = len(all_metrics)
            n_cols = 3
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            fig.suptitle('Metrics Over Time (All Available)', fontsize=16, fontweight='bold')
            
            colors = {'baseline': '#1f77b4', 'actuated': '#ff7f0e', 'skrl_dqn': '#2ca02c'}
            
            for i, metric in enumerate(all_metrics):
                row, col = i // n_cols, i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                for sim_type, df in self.data.items():
                    metric_data = df[df['metric'] == metric].sort_values('time_step')
                    if len(metric_data) > 0:
                        ax.plot(metric_data['time_step'], metric_data['value'], 
                               label=sim_type, color=colors.get(sim_type, 'gray'), linewidth=2)
                
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(all_metrics), n_rows * n_cols):
                row, col = i // n_cols, i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.set_visible(False)
            
            plt.tight_layout()
            plt.savefig(save_path / 'time_series.png', dpi=300, bbox_inches='tight')
            print(f"📈 Saved time series plot: {save_path / 'time_series.png'}")
        
        # 3. Reward plot (if available)
        if 'skrl_dqn' in self.data:
            reward_data = self.data['skrl_dqn'][self.data['skrl_dqn']['metric'] == 'reward']
            if len(reward_data) > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(reward_data['time_step'], reward_data['value'], 
                       color='red', linewidth=2, marker='o', markersize=4)
                ax.set_title('SKRL DQN Reward Over Time', fontsize=14, fontweight='bold')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Reward')
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(reward_data['time_step'], reward_data['value'], 1)
                p = np.poly1d(z)
                ax.plot(reward_data['time_step'], p(reward_data['time_step']), 
                       "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.4f})')
                ax.legend()
                
                plt.tight_layout()
                plt.savefig(save_path / 'reward_analysis.png', dpi=300, bbox_inches='tight')
                print(f"🎯 Saved reward plot: {save_path / 'reward_analysis.png'}")
        
        plt.show()
    
    def generate_report(self, save_path: str = None) -> str:
        """Generate a comprehensive text report."""
        report = []
        report.append("🚦 TRAFFIC SIMULATION METRICS ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Data overview
        report.append("\n📊 DATA OVERVIEW:")
        for sim_type, df in self.data.items():
            unique_metrics = df['metric'].unique()
            time_range = f"{df['time_step'].min()}-{df['time_step'].max()}"
            report.append(f"  {sim_type.upper()}: {len(df)} records, {len(unique_metrics)} metrics, time steps {time_range}")
            report.append(f"    Available metrics: {', '.join(sorted(unique_metrics))}")
        
        # Common vs All metrics info
        common_metrics = self.get_common_metrics()
        all_metrics = self.get_all_available_metrics()
        report.append(f"\n🔄 METRIC AVAILABILITY:")
        report.append(f"  Common metrics (all simulations): {', '.join(common_metrics) if common_metrics else 'None'}")
        report.append(f"  All available metrics: {', '.join(all_metrics)}")
        dqn_only = set(all_metrics) - set(common_metrics)
        if dqn_only:
            report.append(f"  DQN-specific metrics: {', '.join(dqn_only)}")
        
        # Summary statistics
        summary = self.get_metric_summary()
        report.append(f"\n📈 SUMMARY STATISTICS (ALL METRICS):")
        for metric in all_metrics:
            metric_data = summary[summary['metric'] == metric]
            if len(metric_data) > 0:
                report.append(f"\n  {metric.upper().replace('_', ' ')}:")
                for _, row in metric_data.iterrows():
                    report.append(f"    {row['simulation_type']:>12}: Sum={row['sum']:>10.2f}, "
                                f"Mean={row['mean']:>8.2f}, Std={row['std']:>6.2f}, "
                                f"Range=[{row['min']:>6.2f}, {row['max']:>6.2f}]")
        
        # Summary table
        summary_table = self.get_summary_table()
        if not summary_table.empty:
            report.append(f"\n📋 SUMMARY TABLE (Sum/Avg/Min/Max):")
            # Create formatted table
            for _, row in summary_table.iterrows():
                metric = row['metric']
                report.append(f"\n  {metric.upper().replace('_', ' ')}:")
                
                for sim_type in self.simulation_types:
                    if f'{sim_type}_sum' in row and not pd.isna(row[f'{sim_type}_sum']):
                        report.append(f"    {sim_type:>12}: Sum={row[f'{sim_type}_sum']:>10.2f}, "
                                    f"Avg={row[f'{sim_type}_avg']:>8.2f}, "
                                    f"Min={row[f'{sim_type}_min']:>8.2f}, "
                                    f"Max={row[f'{sim_type}_max']:>8.2f}")
        
        # Performance comparison (only common metrics)
        comparison = self.compare_performance()
        if comparison:
            report.append(f"\n🏆 PERFORMANCE COMPARISON (COMMON METRICS ONLY):")
            report.append("  (Lower is better except for outflow and density)")
            report.append(f"  Comparing: {', '.join(comparison.keys())}")
            
            for metric, data in comparison.items():
                direction = "↓" if data['lower_is_better'] else "↑"
                report.append(f"\n  {metric.upper().replace('_', ' ')} {direction}:")
                report.append(f"    Best:  {data['best_simulation']:>12} = {data['best_value']:>8.2f}")
                report.append(f"    Worst: {data['worst_simulation']:>12} = {data['worst_value']:>8.2f}")
                report.append(f"    Improvement: {data['improvement_pct']:>6.1f}%")
        else:
            report.append(f"\n🏆 PERFORMANCE COMPARISON:")
            report.append("  ⚠️  No common metrics found - cannot compare performance across all simulation types")
            report.append("  Consider running simulations with the same metrics for comparison")
        
        # Time series analysis
        time_analysis = self.analyze_time_series()
        report.append(f"\n⏱️  TIME SERIES TRENDS:")
        for metric, sims in time_analysis.items():
            report.append(f"\n  {metric.upper().replace('_', ' ')}:")
            for sim_type, data in sims.items():
                trend = "↗️" if data['trend_correlation'] > 0.1 else "↘️" if data['trend_correlation'] < -0.1 else "➡️"
                report.append(f"    {sim_type:>12} {trend}: {data['change_pct']:>6.1f}% change "
                            f"(correlation: {data['trend_correlation']:>5.3f})")
        
        # Reward analysis (DQN only)
        reward_analysis = self.calculate_reward_analysis()
        if reward_analysis:
            report.append(f"\n🎯 REWARD ANALYSIS (SKRL DQN):")
            report.append(f"    Total Reward: {reward_analysis['total_reward']:>8.2f}")
            report.append(f"    Average Reward: {reward_analysis['average_reward']:>6.2f}")
            report.append(f"    Reward Range: [{reward_analysis['min_reward']:>6.2f}, {reward_analysis['max_reward']:>6.2f}]")
            report.append(f"    Stability (std): {reward_analysis['reward_stability']:>6.3f}")
            trend_desc = "improving" if reward_analysis['reward_trend'] > 0 else "declining" if reward_analysis['reward_trend'] < 0 else "stable"
            report.append(f"    Trend: {trend_desc} (correlation: {reward_analysis['reward_trend']:>5.3f})")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        
        # Find overall best performer
        metric_scores = {}
        for sim_type in self.simulation_types:
            if sim_type in self.data:
                metric_scores[sim_type] = 0
                
        for metric, data in comparison.items():
            if data['best_simulation'] in metric_scores:
                metric_scores[data['best_simulation']] += 1
        
        if metric_scores:
            best_overall = max(metric_scores, key=metric_scores.get)
            report.append(f"  • {best_overall.upper()} wins in {metric_scores[best_overall]} out of {len(comparison)} metrics")
            
            if best_overall == 'skrl_dqn':
                report.append("  • Reinforcement learning shows promising results")
                if reward_analysis and reward_analysis['reward_trend'] > 0:
                    report.append("  • DQN is still learning (positive reward trend)")
                elif reward_analysis and reward_analysis['reward_trend'] < 0:
                    report.append("  • Consider adjusting DQN hyperparameters (negative reward trend)")
            elif best_overall == 'actuated':
                report.append("  • Actuated control performs well with current traffic patterns")
            else:
                report.append("  • Consider optimizing adaptive control methods")
        
        report.append(f"\n" + "=" * 60)
        report.append(f"Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"📝 Report saved to: {save_path}")
        
        return report_text

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='Analyze traffic simulation metrics from CSV files')
    parser.add_argument('--path', type=str, default='models/model_15', 
                       help='Path to directory containing CSV files')
    parser.add_argument('--baseline', type=str, help='Baseline metrics CSV file')
    parser.add_argument('--actuated', type=str, help='Actuated metrics CSV file')
    parser.add_argument('--skrl', type=str, help='SKRL DQN metrics CSV file')
    parser.add_argument('--output', type=str, help='Output directory for plots and reports')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--report', type=str, help='Save report to specified file')
    parser.add_argument('--csv', action='store_true', help='Save summary statistics to CSV files')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TrafficMetricsAnalyzer(args.path)
    
    try:
        # Load data
        print("Loading CSV files...")
        analyzer.load_data(args.baseline, args.actuated, args.skrl)
        
        # Generate report
        print("\nGenerating analysis report...")
        report = analyzer.generate_report(args.report)
        print("\n" + report)
        
        # Save CSV summaries if requested
        if args.csv:
            print("\nSaving summary CSV files...")
            output_path = args.output or args.path
            csv_path = f"{output_path}/metrics_summary.csv" if output_path else "metrics_summary.csv"
            analyzer.save_summary_csv(csv_path)
        
        # Generate plots
        if not args.no_plots:
            print("\nGenerating plots...")
            analyzer.generate_plots(args.output or args.path)
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
