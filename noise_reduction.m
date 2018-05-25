clear all;
close all;

%% Setup
% Declare parameters
alpha_s = .3; % Speech PSD smoothing parameter
alpha_n = .3; % Noise PSD smoothing parameter

ms_window_length = 5;  % How many windows to use in noise PSD estimation
noise_bias = 0;      % bias value in noise PSD
noise_level = 1.;

% Read in data
[speech, fs] = audioread('Audio_files/clean_speech.wav');
[noise, ~] = audioread('Audio_files/speech_shaped_noise.wav');

fprintf('Now playing noisy audio\n');
audio = speech + noise_level .* noise(1:length(speech));
soundsc(audio, fs);
pause(floor(length(audio) / fs));

%% Windowing
% Windowing setup
window_length_sec = 20e-3; %20ms
window_length = window_length_sec * fs;

window = hamming(window_length);
num_windows = 2*floor(length(audio) / window_length) - 1;

% Generate matrix of windows
windowed_audio = zeros(num_windows, window_length);
step_size = floor(window_length / 2);

window_index = 1;
for i = 1:num_windows
    from = step_size * i - step_size + 1;
    to = step_size * i + step_size;
    windowed_audio(i,:) = window .* audio(from:to);
end

%% Processing
fprintf('Processing windowed audio\n');
% Transform to frequency domain
transformed_audio = fft(windowed_audio, window_length, 2);

% Periodogram PSD estimate
psd_est = (abs(transformed_audio) .^ 2) ./ window_length;

% Smoothen the PSD estimate with exp moving avg to do minimum statistics
smoothed_psd = filter(1-alpha_n, [1 -alpha_n], psd_est, [], 1);

% Minimum statistics noise_psd_estimate
for i = 1:size(smoothed_psd, 1)
    if i <= ms_window_length
        noise_psd_est(i,:) = min(smoothed_psd(1:i, :), [], 1);
    else
        noise_psd_est(i,:) = min(smoothed_psd(i-ms_window_length:i, :),...
            [], 1);
    end%if
end%for
noise_psd_est = noise_psd_est + noise_bias;

% Estimate speech SNR, decision directed
speech_snr_est = filter(1-alpha_s, [1 -alpha_n],...
    psd_est ./ noise_psd_est, [], 1);

if sum(sum(isinf(speech_psd_est))) > 0
        fprintf('Speech snr estimate contains Inf\n', i);
end%if

% Wiener filter
filtered_audio = (speech_snr_est ./ (speech_snr_est + 1))...
    .* transformed_audio;

%% Recovering
% Transform to time domain
recovered_windowed_audio = real(ifft(filtered_audio, window_length, 2));

% Overlap add
recovered_length = (floor(num_windows / 2) + 1) * window_length;
recovered_audio = zeros(recovered_length, 1);
for i = 1:num_windows
    from = step_size * (i - 1) + 1;
    to = step_size * (i + 1);
    recovered_audio(from:to) = ...
        recovered_audio(from:to) + ...
        recovered_windowed_audio(i, :)';
end

% fprintf('Playing recovered audio\n');
soundsc(recovered_audio, fs);
error = sum((audio(1:recovered_length) - recovered_audio).^2);
fprintf('Sum of squared errors: %.4f\n', error);
% Plot to compare visually
figure();
subplot(2,1,1);
plot(speech);
title('Original speech signal');
subplot(2,1,2);
plot(recovered_audio);
title('Reconstructed audio signal');
