clear all;
close all;

%% Setup
noise_level = 1;
% Read in data
[speech, fs] = audioread('Audio_files/clean_speech.wav');
[noise, ~] = audioread('Audio_files/Speech_shaped_noise.wav');

speech = speech(1:115531);
audio = speech + noise_level .* noise(1:length(speech));

fprintf('Now playing noisy audio\n');
soundsc(audio, fs);
pause(floor(length(audio) / fs) + 1);
% Declare parameters
window_length_sec = 15e-3; %15ms
window_length = window_length_sec * fs;

window = hamming(window_length);
num_windows = 2*floor(length(audio) / window_length) - 1;

alpha_s = .6; % Speech PSD smoothing parameter

ms_window_length = 96;  % How many windows to use in noise PSD estimation
noise_bias = 4e-5;      % bias value in noise PSD

%% Windowing
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

smoothed_psd = zeros(size(psd_est, 1), size(psd_est, 2));
smoothed_psd(1, :) = psd_est(1,:);
noise_psd_est(1, :) = psd_est(1,:);

alpha_n = ones(1,size(smoothed_psd, 2)) .* 0.5;
for i = 2:size(psd_est, 1)
    % Smooth the PSD estimate
    smoothed_psd(i, :) = alpha_n .* smoothed_psd(i-1, :) + ...
        (1 - alpha_n) .* psd_est(i, :);

    % Find minimum from last few frames
    if i <= ms_window_length
        noise_psd_est(i,:) = min(smoothed_psd(1:i, :), [], 1);
    else
        noise_psd_est(i,:) = min(smoothed_psd(i-ms_window_length:i, :),...
            [], 1);
    end%if
    
    % Update smoothing parameter
    alpha_n = 1 ./...
        (1 + (smoothed_psd(i, :) ./ noise_psd_est(i, :) - 1).^2);
    % Clip it between .96 and .3
    alpha_n(alpha_n > .96) = .96;
    alpha_n(alpha_n < .3) = .3;
end%for

% Estimate speech SNR, decision directed
speech_snr_est = filter(1-alpha_s, [1 -alpha_s],...
    (smoothed_psd ./ noise_psd_est - 1), [], 1);

if sum(sum(isinf(speech_snr_est))) > 0
    fprintf('Speech snr estimate contains Inf\n');
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
error = sum((speech(1:recovered_length) - recovered_audio).^2);
fprintf('Sum of squared errors: %.4f\n', error);
% Plot to compare visually
figure();
subplot(2,1,1);
plot(audio);
title('Original noisy signal');
subplot(2,1,2);
plot(recovered_audio);
title('Reconstructed audio signal');
