"""
Compare multiple tour length result files and take the best (smallest) solution for each instance
Supports any number of input files: all_tour_lengthsv1.txt, all_tour_lengthsv2.txt, etc.
"""

import numpy as np
import argparse
import glob
import os

def load_tour_lengths(filepath):
    """
    Load tour lengths from a text file (skipping comment lines)
    
    Args:
        filepath: Path to the tour lengths file
        
    Returns:
        numpy array of tour lengths
    """
    tour_lengths = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comment lines (starting with #)
            if line and not line.startswith('#'):
                try:
                    length = float(line)
                    tour_lengths.append(length)
                except ValueError:
                    print(f"Warning: Could not parse line in {filepath}: {line}")
                    continue
    
    return np.array(tour_lengths)

def find_all_versions(pattern="all_tour_lengthsv*.txt"):
    """
    Find all version files matching the pattern
    
    Args:
        pattern: File pattern to match
        
    Returns:
        sorted list of matching files
    """
    files = glob.glob(pattern)
    if not files:
        return []
    
    # Sort files naturally (v1, v2, v3, ..., v10, v11, ...)
    import re
    def natural_sort_key(text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]
    
    return sorted(files, key=natural_sort_key)

def compare_multiple_files(files, save_best=False, output_file='best_tour_lengths.txt'):
    """
    Compare multiple sets of tour lengths and select the best (smallest) for each instance
    
    Args:
        files: List of file paths to compare
        save_best: Whether to save the best results to a file
        output_file: Output file name for best results
        
    Returns:
        dict with comparison results
    """
    
    if not files:
        raise ValueError("No files provided for comparison")
    
    print(f"Loading {len(files)} tour length files...")
    
    # Load all files
    all_lengths = []
    file_names = []
    
    for i, filepath in enumerate(files):
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found, skipping...")
            continue
            
        lengths = load_tour_lengths(filepath)
        all_lengths.append(lengths)
        file_names.append(os.path.basename(filepath))
        print(f"  File {i+1} ({os.path.basename(filepath)}): {len(lengths)} instances")
    
    if not all_lengths:
        raise ValueError("No valid files could be loaded")
    
    # Check if all files have same number of instances
    num_instances = [len(lengths) for lengths in all_lengths]
    min_instances = min(num_instances)
    max_instances = max(num_instances)
    
    if min_instances != max_instances:
        print(f"Warning: Files have different number of instances ({min_instances} to {max_instances})")
        print(f"Using first {min_instances} instances from each file")
        all_lengths = [lengths[:min_instances] for lengths in all_lengths]
    
    # Convert to numpy array: (num_files, num_instances)
    all_lengths_array = np.array(all_lengths)
    
    # Find best (minimum) for each instance across all files
    best_lengths = np.min(all_lengths_array, axis=0)
    best_file_indices = np.argmin(all_lengths_array, axis=0)
    
    # Calculate statistics for each file
    file_stats = []
    for i, lengths in enumerate(all_lengths):
        file_wins = np.sum(best_file_indices == i)
        mean_length = np.mean(lengths)
        improvement = lengths - best_lengths
        avg_improvement = np.mean(improvement)
        
        file_stats.append({
            'name': file_names[i],
            'mean': mean_length,
            'wins': file_wins,
            'win_pct': file_wins / len(best_lengths) * 100,
            'avg_improvement_given': avg_improvement
        })
    
    # Overall statistics
    best_mean = np.mean(best_lengths)
    best_std = np.std(best_lengths)
    best_min = np.min(best_lengths)
    best_max = np.max(best_lengths)
    
    # Calculate overall improvement
    individual_means = [stats['mean'] for stats in file_stats]
    best_individual_mean = min(individual_means)
    overall_improvement = (best_individual_mean - best_mean) / best_individual_mean * 100
    
    results = {
        'all_lengths': all_lengths_array,
        'best_lengths': best_lengths,
        'best_file_indices': best_file_indices,
        'file_names': file_names,
        'file_stats': file_stats,
        'num_instances': len(best_lengths),
        'num_files': len(all_lengths),
        'best_mean': best_mean,
        'best_std': best_std,
        'best_min': best_min,
        'best_max': best_max,
        'overall_improvement': overall_improvement
    }
    
    # Print detailed comparison
    print("\n" + "="*70)
    print("MULTI-FILE COMPARISON RESULTS")
    print("="*70)
    
    print(f"Total files compared: {len(all_lengths)}")
    print(f"Total instances: {len(best_lengths)}")
    
    print(f"\nIndividual file performance:")
    for i, stats in enumerate(file_stats):
        print(f"  {stats['name']:<20}: mean={stats['mean']:7.4f}, "
              f"wins={stats['wins']:4d} ({stats['win_pct']:5.1f}%)")
    
    print(f"\nBest combination results:")
    print(f"  Best mean: {best_mean:.4f}")
    print(f"  Best std:  {best_std:.4f}")
    print(f"  Best min:  {best_min:.4f}")
    print(f"  Best max:  {best_max:.4f}")
    
    print(f"\nImprovement analysis:")
    best_file_idx = np.argmin(individual_means)
    print(f"  Best individual file: {file_names[best_file_idx]} (mean: {individual_means[best_file_idx]:.4f})")
    print(f"  Best combination improves by: {overall_improvement:.2f}%")
    
    # Show which file contributed most
    file_contributions = [(stats['wins'], stats['name']) for stats in file_stats]
    file_contributions.sort(reverse=True)
    
    print(f"\nFile contributions (instances won):")
    for wins, name in file_contributions[:5]:  # Show top 5
        pct = wins / len(best_lengths) * 100
        print(f"  {name:<20}: {wins:4d} instances ({pct:5.1f}%)")
    
    # Per-instance improvement statistics
    total_improvements = []
    for i, lengths in enumerate(all_lengths):
        improvements = lengths - best_lengths
        total_improvements.extend(improvements)
    
    avg_improvement_per_instance = np.mean(best_lengths - np.mean(all_lengths_array, axis=0))
    
    print(f"\nPer-instance improvement:")
    print(f"  Average improvement per instance: {avg_improvement_per_instance:.4f}")
    print(f"  Max improvement for any instance: {np.max([np.max(lengths - best_lengths) for lengths in all_lengths]):.4f}")
    
    # Save best results if requested
    if save_best:
        with open(output_file, 'w') as f:
            f.write("# Best tour lengths (minimum across all files) in default instance order\n")
            f.write(f"# Total instances: {len(best_lengths)}\n")
            f.write(f"# Source files: {len(file_names)} files\n")
            for name in file_names:
                f.write(f"#   {name}\n")
            f.write(f"# Best mean: {best_mean:.6f}\n")
            f.write(f"# Overall improvement: {overall_improvement:.2f}%\n")
            f.write("#" + "="*50 + "\n")
            
            for i, length in enumerate(best_lengths):
                source_file = file_names[best_file_indices[i]]
                f.write(f"{length:.6f}  # from {source_file}\n")
        
        print(f"\nBest results saved to: {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Compare multiple tour length result files')
    
    parser.add_argument('files', nargs='*', help='Tour length files to compare')
    parser.add_argument('--pattern', type=str, default='all_tour_lengthsv*.txt',
                       help='File pattern to match (default: all_tour_lengthsv*.txt)')
    parser.add_argument('--save_best', action='store_true', 
                       help='Save best results to file')
    parser.add_argument('--output', type=str, default='best_tour_lengths_combined.txt',
                       help='Output file for best results')
    
    args = parser.parse_args()
    
    try:
        # If specific files provided, use them; otherwise use pattern matching
        if args.files:
            files_to_compare = args.files
            print(f"Comparing {len(files_to_compare)} specified files...")
        else:
            files_to_compare = find_all_versions(args.pattern)
            if not files_to_compare:
                print(f"No files found matching pattern: {args.pattern}")
                print("Available files in current directory:")
                all_txt_files = glob.glob("*.txt")
                for f in all_txt_files:
                    if "tour_length" in f:
                        print(f"  {f}")
                return
            print(f"Found {len(files_to_compare)} files matching pattern '{args.pattern}':")
            for f in files_to_compare:
                print(f"  {f}")
        
        results = compare_multiple_files(
            files_to_compare,
            save_best=args.save_best,
            output_file=args.output
        )
        
        print(f"\n🎯 SUMMARY:")
        print(f"   Best mean tour length: {results['best_mean']:.4f}")
        print(f"   Total instances: {results['num_instances']}")
        print(f"   Files compared: {results['num_files']}")
        print(f"   Overall improvement: {results['overall_improvement']:.2f}%")
        
        if args.save_best:
            print(f"   Best results saved to: {args.output}")
            
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
