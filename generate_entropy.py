import matplotlib.pyplot as plt
import numpy as np

# Sample data for 10 images
indices = np.arange(0, 10)
entropy_time_series = np.random.rand(10) * 0.1  # Random entropy values for time series images
entropy_test_set = np.random.rand(10) * 0.1     # Random entropy values for test set images

# Indices split for time series and test set
time_series_indices = indices[:5]
test_set_indices = indices[5:]

# Generate the plot
plt.figure(figsize=(10, 6))
plt.bar(test_set_indices, entropy_test_set[5:], label='Images in The Test Set')
plt.bar(time_series_indices, entropy_time_series[:5], label='Images in The Time-Series')


# Add labels and title
plt.xlabel('Index of Images')
plt.ylabel('Entropy')
plt.title('Beijing')
plt.legend()

# Show the plot
plt.show()
