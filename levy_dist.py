import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable

def sample_levy(alpha, beta, size):
    """Sample from an α-stable Lévy distribution."""
    return levy_stable.rvs(alpha, beta, size=size)

def visualize(data, alpha, beta, bins=300):
    plt.figure(figsize=(10, 6))

    # Histogram with log scale to show heavy tail properly
    plt.hist(data, bins=bins, density=True, alpha=0.7)
    plt.yscale("log")

    plt.title(f"Lévy α-stable Distribution (alpha={alpha}, beta={beta})")
    plt.xlabel("x")
    plt.ylabel("Probability Density (log scale)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize Lévy / α-stable distribution")
    parser.add_argument("--alpha", type=float, default=1.5,
                        help="Stability parameter α (0 < alpha ≤ 2)")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Skewness β (-1 ≤ beta ≤ 1)")
    parser.add_argument("--size", type=int, default=50000,
                        help="Number of samples")
    args = parser.parse_args()

    data = sample_levy(args.alpha, args.beta, args.size)
    visualize(data, args.alpha, args.beta)

if __name__ == "__main__":
    main()
