import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.gridspec as gridspec

# Path to your data
dir_path = "/home/zj2398@mc.cumc.columbia.edu/ICL_EHR/hf_ehr/cache/tokenizers/cookbook_39k/versions/2025-04-08_00-21-55/datasets"

# Create a figure with subplots - one for histogram, one for CDF
# Use a clean style for better visibility
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    # Fallback for newer matplotlib versions
    plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

# You can replace datasets with ["train","val","test"]
datasets = ["train"]
for dataset in datasets:
    # Load data
    json_file = os.path.join(dir_path, dataset, "seq_length_per_patient.json")
    with open(json_file, 'r') as f:
        seq_length_per_patient = json.load(f)["seq_lengths"]
    
    data_array = np.array(seq_length_per_patient)
    total_count = len(data_array)
    
    # Calculate thresholds and percentiles
    thresholds = [2048, 4096, 8192]
    for t in thresholds:
        count = np.sum(data_array < t)
        print(f"Sequences shorter than {t}: {count / total_count * 100:.2f}%")
    
    percentiles = [50, 90, 95, 99]
    percentile_values = np.percentile(data_array, percentiles)
    for p, val in zip(percentiles, percentile_values):
        print(f"{p}th Percentile: {val:.0f}")
    
    print(f"For {dataset} dataset: the maximum length is {max(data_array)}")
    
    # Histogram with percentage on y-axis
    ax1 = plt.subplot(gs[0])
    
    # Custom bins to better visualize both small and large values
    # Create more bins in the 0-1000 range for detailed view
    small_bins = np.linspace(0, 1000, 50)  # 50 bins for 0-1000 range
    medium_bins = np.linspace(1000, 10000, 30)  # 30 bins from 1000-10000
    large_bins = np.logspace(np.log10(10000), np.log10(data_array.max()+1000), 20)  # Log-spaced bins for the tail
    
    # Combine the bins and remove duplicates
    custom_bins = np.unique(np.concatenate([small_bins, medium_bins, large_bins]))
    
    # Create histogram with percentage
    counts, bins, patches = ax1.hist(
        data_array, 
        bins=custom_bins,
        density=True,  # Normalize to probability density
        alpha=0.7,
        color='skyblue',
        edgecolor='black'
    )
    
    # Convert density to percentage (multiply by 100)
    ax1.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    
    # Add vertical lines for thresholds
    colors = ['red', 'orange', 'green']
    for t, c in zip(thresholds, colors):
        ax1.axvline(x=t, color=c, linestyle='--', linewidth=2, 
                   label=f'Threshold: {t}')
    
    # Set log scale for x-axis to better visualize the long tail
    ax1.set_xscale('log')
    
    # Add a secondary plot with linear scale for better visualization of 0-1000 range
    # Create an inset axes
    axins = ax1.inset_axes([0.05, 0.5, 0.4, 0.45])  # [x, y, width, height] in normalized coordinates
    
    # Plot the histogram for 0-1000 range with linear scale
    small_bins = np.linspace(0, 1000, 100)  # 100 bins for even more detail
    axins.hist(data_array, bins=small_bins, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    
    # Set the limits to focus on 0-1000 range
    axins.set_xlim(0, 1000)
    
    # Format the inset
    axins.set_title("Detailed view: 0-1000 tokens", fontsize=10)
    axins.grid(True, alpha=0.3)
    axins.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    
    # Add labels and title
    ax1.set_title(f"Distribution of Sequence Lengths ({dataset} dataset)", fontsize=16)
    ax1.set_xlabel("Sequence Length (tokens)", fontsize=14)
    ax1.set_ylabel("Percentage of Patients", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Annotate max value
    ax1.annotate(f'Max: {max(data_array)}', 
                xy=(max(data_array), 0), 
                xytext=(max(data_array)*0.7, max(counts)*0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)
    
    # CDF plot (second subplot)
    ax2 = plt.subplot(gs[1])
    
    # Calculate the CDF
    sorted_data = np.sort(data_array)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Plot the CDF
    ax2.plot(sorted_data, cdf * 100, 'b-', linewidth=2, label='CDF')
    
    # Add vertical lines for thresholds on CDF plot
    for t, c in zip(thresholds, colors):
        ax2.axvline(x=t, color=c, linestyle='--', linewidth=2)
        # Find the CDF value at this threshold
        idx = np.searchsorted(sorted_data, t)
        if idx < len(sorted_data):
            cdf_at_threshold = cdf[idx] * 100
            ax2.plot([t], [cdf_at_threshold], 'o', color=c, markersize=8)
            ax2.annotate(f'{cdf_at_threshold:.1f}%', 
                        xy=(t, cdf_at_threshold),
                        xytext=(t*1.1, cdf_at_threshold),
                        fontsize=10)
    
    # Set log scale for x-axis to match the histogram
    ax2.set_xscale('log')
    
    # Add labels
    ax2.set_xlabel("Sequence Length (tokens)", fontsize=14)
    ax2.set_ylabel("Cumulative Percentage", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Cumulative Distribution of Sequence Lengths", fontsize=16)
    
    # Set y-axis to percentage
    ax2.yaxis.set_major_formatter(PercentFormatter())
    
plt.tight_layout()
plt.savefig("sequence_length_distribution.png", dpi=300, bbox_inches='tight')
plt.show()