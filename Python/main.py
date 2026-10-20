
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# STEP 1: Locate the project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# STEP 2: Define dataset paths
normal_path = os.path.join(
    PROJECT_DIR, "Dataset", "Normal", "97.mat"
)

fault_path = os.path.join(
    PROJECT_DIR, "Dataset", "Inner Race", "105.mat"
)

# STEP 3: Check whether dataset files exist
if not os.path.isfile(normal_path):
    raise FileNotFoundError(
        f"Normal bearing dataset not found:\n{normal_path}\n"
        "Check that 97.mat is inside Dataset/Normal/"
    )

if not os.path.isfile(fault_path):
    raise FileNotFoundError(
        f"Fault bearing dataset not found:\n{fault_path}\n"
        "Check that 105.mat is inside Dataset/Inner Race/"
    )


# STEP 4: Load MATLAB dataset files
normal_data = loadmat(normal_path)
fault_data = loadmat(fault_path)

# STEP 5: Extract drive-end vibration signals
normal_key = "X097_DE_time"
fault_key = "X105_DE_time"

if normal_key not in normal_data:
    raise KeyError(
        f"Variable '{normal_key}' was not found in 97.mat. "
        f"Available variables: "
        f"{[key for key in normal_data.keys() if not key.startswith('__')]}"
    )

if fault_key not in fault_data:
    raise KeyError(
        f"Variable '{fault_key}' was not found in 105.mat. "
        f"Available variables: "
        f"{[key for key in fault_data.keys() if not key.startswith('__')]}"
    )

normal_signal = np.asarray(
    normal_data[normal_key], dtype=np.float64
).flatten()

fault_signal = np.asarray(
    fault_data[fault_key], dtype=np.float64
).flatten()

# STEP 6: Display basic signal information
print("=" * 60)
print("BEARING FAULT DETECTION - SIGNAL ANALYSIS")
print("=" * 60)

print("\nNormal Bearing Signal")
print("---------------------")
print("Number of samples:", len(normal_signal))
print("Minimum amplitude:", np.min(normal_signal))
print("Maximum amplitude:", np.max(normal_signal))
print("Mean amplitude:", np.mean(normal_signal))

print("\nInner-Race Fault Signal")
print("-----------------------")
print("Number of samples:", len(fault_signal))
print("Minimum amplitude:", np.min(fault_signal))
print("Maximum amplitude:", np.max(fault_signal))
print("Mean amplitude:", np.mean(fault_signal))

print("\nFirst 10 normal signal samples:")
print(normal_signal[:10])

print("\nFirst 10 fault signal samples:")
print(fault_signal[:10])

# STEP 7: Set sampling frequency and plotting parameters
Fs = 12000  # Sampling frequency in Hz
samples_to_plot = 2000

# Use available samples safely
normal_count = min(samples_to_plot, len(normal_signal))
fault_count = min(samples_to_plot, len(fault_signal))

normal_time = np.arange(normal_count) / Fs
fault_time = np.arange(fault_count) / Fs

# STEP 8: Create Results folder
results_dir = os.path.join(PROJECT_DIR, "Results")
os.makedirs(results_dir, exist_ok=True)

# STEP 9: Plot time-domain vibration signals
plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
plt.plot(
    normal_time,
    normal_signal[:normal_count],
    linewidth=0.8
)
plt.title("Normal Bearing - Drive-End Vibration")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.5)

plt.subplot(2, 1, 2)
plt.plot(
    fault_time,
    fault_signal[:fault_count],
    linewidth=0.8
)
plt.title("Inner-Race Fault - Drive-End Vibration")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.5)
plt.tight_layout()

# STEP 10: Save the comparison graph
plot_path = os.path.join(
    results_dir,
    "time_domain_comparison.png"
)
plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)
print("\nTime-domain graph saved successfully:")
print(plot_path)

# STEP 11: Display the graph
plt.show()

print("\nSignal loading and time-domain visualization completed.")
