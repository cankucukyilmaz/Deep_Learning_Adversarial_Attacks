import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_imbalance(processed_path, save_plot_path=None):
    df = pd.read_csv(processed_path, index_col=0)
    
    # Calculate counts
    # Summing the columns gives the number of 1s (since 0s don't add up)
    positive_counts = df.sum().sort_values(ascending=False)
    total_samples = len(df)
    
    # Calculate percentage
    positive_ratio = (positive_counts / total_samples) * 100
    
    # Plotting
    plt.figure(figsize=(15, 8))
    sns.barplot(x=positive_ratio.index, y=positive_ratio.values, palette="viridis")
    
    plt.axhline(50, color='r', linestyle='--', alpha=0.5, label='Perfect Balance')
    plt.xticks(rotation=90)
    plt.ylabel('Percentage of Positive Samples (%)')
    plt.title('CelebA Attribute Distribution (Imbalance Check)')
    plt.legend()
    plt.tight_layout()
    
    if save_plot_path:
        plt.savefig(save_plot_path)
        print(f"Plot saved to {save_plot_path}")
    else:
        plt.show()

    # Print the specific stats for your group's chosen attribute
    # Since I don't know which one you picked, I'll print the top 5 most imbalanced
    print("\nMost imbalanced attributes (lowest positive presence):")
    print(positive_ratio.tail(5))

if __name__ == "__main__":
    PROCESSED_FILE = 'data/processed/celeba_attrs_clean.csv'
    PLOT_FILE = 'plots/attribute_imbalance.png'
    os.makedirs('plots', exist_ok=True)
    
    plot_imbalance(PROCESSED_FILE, PLOT_FILE)