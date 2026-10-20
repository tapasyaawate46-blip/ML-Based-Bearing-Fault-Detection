clear;
clc;
close all;

%% 1. Load bearing vibration data
normalData = load('97.mat');
faultData  = load('105.mat');

normalSignal = double(normalData.X097_DE_time(:));
faultSignal  = double(faultData.X105_DE_time(:));

%% 2. Sampling frequency
Fs = 12000;

fprintf('Normal samples: %d\n', length(normalSignal));
fprintf('Fault samples: %d\n', length(faultSignal));

%% 3. Select samples for plotting
nPlot = min(2000, min(length(normalSignal), ...
    length(faultSignal)));

t = (0:nPlot-1)' / Fs;

%% 4. Plot time-domain signals
figure('Name', 'Bearing Vibration Analysis');

subplot(2,1,1);
plot(t, normalSignal(1:nPlot));
title('Normal Bearing Vibration');
xlabel('Time (seconds)');
ylabel('Amplitude');
grid on;

subplot(2,1,2);
plot(t, faultSignal(1:nPlot));
title('Inner-Race Fault Vibration');
xlabel('Time (seconds)');
ylabel('Amplitude');
grid on;

%% 5. FFT analysis
windowLength = 2048;

% Select one segment from each signal
xN = normalSignal(1:windowLength);
xF = faultSignal(1:windowLength);

% Remove the mean value (DC component)
xN = xN - mean(xN);
xF = xF - mean(xF);

% Create a Hann window manually
w = 0.5 - 0.5*cos(2*pi*(0:windowLength-1)' ...
    /(windowLength-1));

% Apply window and calculate FFT
XN = fft(xN .* w);
XF = fft(xF .* w);

% Keep the positive-frequency half
nHalf = floor(windowLength/2) + 1;
freq = (0:nHalf-1)' * Fs / windowLength;

% Calculate single-sided amplitude spectra
ampN = abs(XN(1:nHalf)) / sum(w);
ampF = abs(XF(1:nHalf)) / sum(w);

% Correct the positive-frequency amplitudes
ampN(2:end-1) = 2*ampN(2:end-1);
ampF(2:end-1) = 2*ampF(2:end-1);

% Plot FFT comparison
figure('Name','FFT Comparison');

subplot(2,1,1);
plot(freq,ampN);
title('Normal Bearing - FFT Spectrum');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
grid on;
xlim([0 Fs/2]);

subplot(2,1,2);
plot(freq,ampF);
title('Inner-Race Fault - FFT Spectrum');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
grid on;
xlim([0 Fs/2]);

%% 6. Feature Extraction for Machine Learning
windowLength = 2048;
overlapLength = 1024;
step = windowLength - overlapLength;

% Create a Hann window without requiring a toolbox
w = 0.5 - 0.5*cos(2*pi*(0:windowLength-1)' ...
    /(windowLength-1));

nHalf = floor(windowLength/2) + 1;
freq = (0:nHalf-1)' * Fs / windowLength;

signals = {normalSignal, faultSignal};
classLabels = [0, 1];

features = [];
labels = [];

for c = 1:2

    x = signals{c};
    starts = 1:step:(length(x)-windowLength+1);

    for k = 1:length(starts)

        segment = x(starts(k):starts(k)+windowLength-1);

        % Remove the mean
        segment = segment - mean(segment);

        % Time-domain features
        rmsValue = sqrt(mean(segment.^2));
        stdValue = std(segment);
        peakValue = max(abs(segment));

        % Frequency-domain feature
        windowedSignal = segment .* w;
        spectrum = fft(windowedSignal);
        magnitude = abs(spectrum(1:nHalf));

        % Ignore DC when finding dominant frequency
        magnitude(1) = 0;
        [~, peakIndex] = max(magnitude);
        dominantFreq = freq(peakIndex);

        % Window-corrected mean-square feature
        spectralEnergy = sum(windowedSignal.^2) / sum(w.^2);

        features(end+1,:) = [rmsValue, stdValue, ...
            peakValue, dominantFreq, spectralEnergy];

        labels(end+1,1) = classLabels(c);
    end
end

% Create and export feature table
featureTable = array2table(features, ...
    'VariableNames', {'RMS','StdDev','PeakAmplitude', ...
    'DominantFrequency','SpectralEnergy'});

featureTable.Condition = labels;

writetable(featureTable, 'bearing_features.csv');

fprintf('Feature extraction completed.\n');
fprintf('Total windows: %d\n', height(featureTable));
fprintf('CSV saved as bearing_features.csv\n');
