import matplotlib.pyplot as plt
import numpy as np

indices = np.arange(0, 10)
entropy_time_series = np.random.rand(10) * 0.1  
entropy_test_set = np.random.rand(10) * 0.1    

time_series_indices = indices[:5]
test_set_indices = indices[5:]

plt.figure(figsize=(10, 6))
plt.bar(test_set_indices, entropy_test_set[5:], label='Images in The Test Set')
plt.bar(time_series_indices, entropy_time_series[:5], label='Images in The Time-Series')


plt.xlabel('Index of Images')
plt.ylabel('Entropy')
plt.title('Beijing')
plt.legend()

plt.show()
