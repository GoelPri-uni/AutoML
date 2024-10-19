import numpy as np 
import matplotlib.pyplot as plt

iterations = [1, 2, 3, 4]  # Number of iterations (successive halving rounds)
configurations = ['Config 1', 'Config 2', 'Config 3', 'Config 4', 'Config 5']  # Example configurations


performance = [
    [0.65, 0.72, 0.80, 0.82],  # Config 1
    [0.60, 0.68, 0.75, np.nan],  # Config 2 (eliminated after iteration 3)
    [0.62, 0.70, 0.77, np.nan],  # Config 3 (eliminated after iteration 3)
    [0.58, 0.66, np.nan, np.nan],  # Config 4 (eliminated after iteration 2)
    [0.57, np.nan, np.nan, np.nan]  # Config 5 (eliminated after iteration 1)
]

# Plotting
plt.figure(figsize=(10, 6))

# Plot each configuration's performance over iterations
for i, config in enumerate(configurations):
    plt.plot(iterations, performance[i], label=config, marker='o', linestyle='--')

# Adding labels, title, and legend
plt.xlabel('Iterations (Successive Halving Rounds)')
plt.ylabel('Performance (e.g., Accuracy)')
plt.title('Successive Halving Algorithm Performance')
plt.xticks(iterations)  # Set iteration points on the x-axis
plt.yticks(np.arange(0.5, 1.0, 0.05))  # Set reasonable y-tick range for performance

# Add legend to show which line corresponds to which configuration
plt.legend(title="Configurations")

# Show plot
plt.grid(True)
plt.show()


