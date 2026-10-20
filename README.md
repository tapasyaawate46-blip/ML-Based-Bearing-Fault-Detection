# ML-Based Bearing Fault Detection from FFT Features

## Overview
This project implements a machine learning-based bearing fault detection system using vibration signals and Digital Signal Processing (DSP) techniques. 
FFT analysis and statistical feature extraction are used to distinguish between a normal bearing and a bearing with an inner-race fault.
The system uses MATLAB for signal processing and feature extraction, and Python with a Random Forest classifier for machine learning and evaluation.

## Objectives
- Analyze bearing vibration signals in the time and frequency domains.
- Apply Fast Fourier Transform (FFT) to study frequency components.
- Extract useful features from vibration signals.
- Classify bearing conditions using a Random Forest model.
- Evaluate the model using a confusion matrix and classification metrics.

## Technologies Used
- MATLAB
- Python
- NumPy
- SciPy
- Pandas
- Matplotlib
- Scikit-learn

## Project Structure
ML-Based-Bearing-Fault-Detection/
├── Matlab/
│   └── bearing.m
├── Python/
│   ├── main.py
│   └── train_model.py
└── Results/
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── model_results.txt
    └── time_domain_comparison.png

## Methodology
1. **Signal loading:** Load normal and inner-race fault vibration signals from MATLAB `.mat` files.
2. **Time-domain analysis:** Plot vibration signals to compare their amplitude variations.
3. **FFT analysis:** Remove the mean, apply a Hann window, and calculate the frequency spectrum.
4. **Feature extraction:** Extract five features using 2048-sample windows with 1024-sample overlap:
   - RMS
   - Standard Deviation
   - Peak Amplitude
   - Dominant Frequency
   - Spectral Energy
5. **Feature export:** Save the extracted features and class labels to `bearing_features.csv`.
6. **Model training:** Train a Random Forest classifier using the extracted features.
7. **Evaluation:** Generate a confusion matrix, classification report, and feature-importance plot.

## Dataset
The project uses the Case Western Reserve University (CWRU) Bearing Data Center dataset.
Dataset website: https://engineering.case.edu/bearingdatacenter/welcome
The raw dataset files are not included in this repository. Download the required normal and inner-race fault recordings separately.

## Results
The current test run achieved **100% accuracy on 72 test samples**:
- Normal samples correctly classified: 48
- Inner-race fault samples correctly classified: 24
- Misclassified samples: 0
The feature-importance analysis ranked Dominant Frequency as the most important feature in this run.

**Note:** The training and testing windows originate from the same source recordings. Therefore, this result does not establish generalization to independent recordings, and further testing is required.

## How to Run
1. Download the required CWRU `.mat` files.
2. Run `bearing.m` in MATLAB to perform signal analysis, extract features, and generate `bearing_features.csv`.
3. Place the generated CSV where `train_model.py` expects it: inside the `Dataset` folder alongside the Python script's project directory.
4. Install the required Python libraries.
5. Run `main.py` for Python-based signal visualization, or run `train_model.py` to train and evaluate the classifier.

Refer to the paths defined in the scripts when arranging the input files.

## Future Scope
- Evaluate the model on independent bearing recordings.
- Include additional bearing fault types and operating conditions.
- Compare Random Forest with other machine learning algorithms.
- Explore real-time bearing monitoring using vibration sensors.
