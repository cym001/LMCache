#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Analyze multi-round conversation test results
"""

import argparse
import pandas as pd
import sys
import numpy as np


def analyze_results(csv_file: str):
    """Analyze and print detailed statistics from multi-round conversation test results"""
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)
    
    if len(df) == 0:
        print("Error: CSV file is empty")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("MULTI-ROUND CONVERSATION TEST - DETAILED ANALYSIS")
    print("=" * 80)
    
    # Overall statistics
    print("\n📊 OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total turns: {len(df)}")
    print(f"Total conversation rounds: {df['round_id'].nunique()}")
    print(f"Instances: {', '.join(df['instance_id'].unique())}")
    
    # Split by instance
    df_instance1 = df[df['instance_id'] == 'instance1']
    df_instance2 = df[df['instance_id'] == 'instance2']
    
    # Instance 1 statistics
    if len(df_instance1) > 0:
        print("\n🔵 INSTANCE 1 (Original Order)")
        print("-" * 80)
        print(f"  Total turns: {len(df_instance1)}")
        print(f"  Conversation rounds: {df_instance1['round_id'].nunique()}")
        print(f"  Average turns per conversation: {df_instance1.groupby('round_id')['turn_id'].max().mean():.1f}")
        print(f"\n  Performance Metrics:")
        print(f"    Average TTFT: {df_instance1['ttft'].mean():.4f}s")
        print(f"    Median TTFT: {df_instance1['ttft'].median():.4f}s")
        print(f"    Min TTFT: {df_instance1['ttft'].min():.4f}s")
        print(f"    Max TTFT: {df_instance1['ttft'].max():.4f}s")
        print(f"    Std Dev TTFT: {df_instance1['ttft'].std():.4f}s")
        print(f"\n    Average Generation Time: {df_instance1['generation_time'].mean():.4f}s")
        print(f"    Average Total Time: {df_instance1['total_time'].mean():.4f}s")
        print(f"\n    Average Prompt Tokens: {df_instance1['prompt_tokens'].mean():.1f}")
        print(f"    Average Generation Tokens: {df_instance1['generation_tokens'].mean():.1f}")
        
        # Token throughput
        avg_prompt_tokens = df_instance1['prompt_tokens'].mean()
        avg_ttft = df_instance1['ttft'].mean()
        if avg_ttft > 0:
            throughput = avg_prompt_tokens / avg_ttft
            print(f"    Prefill Throughput: {throughput:.2f} tokens/s")
    
    # Instance 2 statistics
    if len(df_instance2) > 0:
        print("\n🟢 INSTANCE 2 (Shuffled Order)")
        print("-" * 80)
        print(f"  Total turns: {len(df_instance2)}")
        print(f"  Conversation rounds: {df_instance2['round_id'].nunique()}")
        print(f"  Average turns per conversation: {df_instance2.groupby('round_id')['turn_id'].max().mean():.1f}")
        print(f"\n  Performance Metrics:")
        print(f"    Average TTFT: {df_instance2['ttft'].mean():.4f}s")
        print(f"    Median TTFT: {df_instance2['ttft'].median():.4f}s")
        print(f"    Min TTFT: {df_instance2['ttft'].min():.4f}s")
        print(f"    Max TTFT: {df_instance2['ttft'].max():.4f}s")
        print(f"    Std Dev TTFT: {df_instance2['ttft'].std():.4f}s")
        print(f"\n    Average Generation Time: {df_instance2['generation_time'].mean():.4f}s")
        print(f"    Average Total Time: {df_instance2['total_time'].mean():.4f}s")
        print(f"\n    Average Prompt Tokens: {df_instance2['prompt_tokens'].mean():.1f}")
        print(f"    Average Generation Tokens: {df_instance2['generation_tokens'].mean():.1f}")
        
        # Token throughput
        avg_prompt_tokens = df_instance2['prompt_tokens'].mean()
        avg_ttft = df_instance2['ttft'].mean()
        if avg_ttft > 0:
            throughput = avg_prompt_tokens / avg_ttft
            print(f"    Prefill Throughput: {throughput:.2f} tokens/s")
    
    # Comparison
    if len(df_instance1) > 0 and len(df_instance2) > 0:
        print("\n📈 COMPARISON (Instance 2 vs Instance 1)")
        print("-" * 80)
        
        # TTFT comparison
        ttft_1 = df_instance1['ttft'].mean()
        ttft_2 = df_instance2['ttft'].mean()
        ttft_diff = ttft_2 - ttft_1
        ttft_pct = (ttft_diff / ttft_1) * 100 if ttft_1 > 0 else 0
        ttft_ratio = ttft_2 / ttft_1 if ttft_1 > 0 else 0
        
        print(f"  TTFT:")
        print(f"    Instance 1: {ttft_1:.4f}s")
        print(f"    Instance 2: {ttft_2:.4f}s")
        print(f"    Difference: {ttft_diff:+.4f}s ({ttft_pct:+.2f}%)")
        print(f"    Ratio: {ttft_ratio:.2f}x")
        
        # Total time comparison
        total_1 = df_instance1['total_time'].mean()
        total_2 = df_instance2['total_time'].mean()
        total_diff = total_2 - total_1
        total_pct = (total_diff / total_1) * 100 if total_1 > 0 else 0
        total_ratio = total_2 / total_1 if total_1 > 0 else 0
        
        print(f"\n  Total Time:")
        print(f"    Instance 1: {total_1:.4f}s")
        print(f"    Instance 2: {total_2:.4f}s")
        print(f"    Difference: {total_diff:+.4f}s ({total_pct:+.2f}%)")
        print(f"    Ratio: {total_ratio:.2f}x")
        
        # Generation time comparison
        gen_1 = df_instance1['generation_time'].mean()
        gen_2 = df_instance2['generation_time'].mean()
        gen_diff = gen_2 - gen_1
        gen_pct = (gen_diff / gen_1) * 100 if gen_1 > 0 else 0
        
        print(f"\n  Generation Time:")
        print(f"    Instance 1: {gen_1:.4f}s")
        print(f"    Instance 2: {gen_2:.4f}s")
        print(f"    Difference: {gen_diff:+.4f}s ({gen_pct:+.2f}%)")
    
    # Turn-wise analysis (first turn vs subsequent turns)
    print("\n🔄 TURN-WISE ANALYSIS")
    print("-" * 80)
    
    for instance_name, df_inst in [("Instance 1", df_instance1), ("Instance 2", df_instance2)]:
        if len(df_inst) > 0:
            df_first_turn = df_inst[df_inst['turn_id'] == 1]
            df_subsequent = df_inst[df_inst['turn_id'] > 1]
            
            print(f"\n  {instance_name}:")
            if len(df_first_turn) > 0:
                print(f"    First turn average TTFT: {df_first_turn['ttft'].mean():.4f}s")
            if len(df_subsequent) > 0:
                print(f"    Subsequent turns average TTFT: {df_subsequent['ttft'].mean():.4f}s")
            if len(df_first_turn) > 0 and len(df_subsequent) > 0:
                improvement = ((df_first_turn['ttft'].mean() - df_subsequent['ttft'].mean()) / 
                              df_first_turn['ttft'].mean() * 100)
                print(f"    Improvement in subsequent turns: {improvement:.2f}%")
    
    # Distribution analysis
    print("\n📊 TTFT DISTRIBUTION")
    print("-" * 80)
    
    for instance_name, df_inst in [("Instance 1", df_instance1), ("Instance 2", df_instance2)]:
        if len(df_inst) > 0:
            print(f"\n  {instance_name}:")
            percentiles = [25, 50, 75, 90, 95, 99]
            for p in percentiles:
                val = np.percentile(df_inst['ttft'], p)
                print(f"    P{p}: {val:.4f}s")
    
    # Top slowest and fastest turns
    print("\n🐢 SLOWEST TURNS (Top 5)")
    print("-" * 80)
    slowest = df.nlargest(5, 'ttft')[['instance_id', 'round_id', 'turn_id', 'ttft', 'total_time', 'prompt_tokens']]
    for idx, row in slowest.iterrows():
        print(f"  {row['instance_id']}, Round {row['round_id']}, Turn {row['turn_id']}: "
              f"TTFT={row['ttft']:.4f}s, Total={row['total_time']:.4f}s, Prompt={row['prompt_tokens']:.0f} tokens")
    
    print("\n🚀 FASTEST TURNS (Top 5)")
    print("-" * 80)
    fastest = df.nsmallest(5, 'ttft')[['instance_id', 'round_id', 'turn_id', 'ttft', 'total_time', 'prompt_tokens']]
    for idx, row in fastest.iterrows():
        print(f"  {row['instance_id']}, Round {row['round_id']}, Turn {row['turn_id']}: "
              f"TTFT={row['ttft']:.4f}s, Total={row['total_time']:.4f}s, Prompt={row['prompt_tokens']:.0f} tokens")
    
    # Conversation length analysis
    print("\n💬 CONVERSATION LENGTH ANALYSIS")
    print("-" * 80)
    
    for instance_name, df_inst in [("Instance 1", df_instance1), ("Instance 2", df_instance2)]:
        if len(df_inst) > 0:
            turns_per_conv = df_inst.groupby('round_id')['turn_id'].max()
            print(f"\n  {instance_name}:")
            print(f"    Min turns in a conversation: {turns_per_conv.min():.0f}")
            print(f"    Max turns in a conversation: {turns_per_conv.max():.0f}")
            print(f"    Average turns per conversation: {turns_per_conv.mean():.1f}")
            print(f"    Median turns per conversation: {turns_per_conv.median():.1f}")
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi-round conversation test results"
    )
    parser.add_argument(
        "csv_file",
        type=str,
        nargs='?',
        default="conversation_results.csv",
        help="CSV file with test results (default: conversation_results.csv)"
    )
    
    args = parser.parse_args()
    analyze_results(args.csv_file)


if __name__ == "__main__":
    main()
