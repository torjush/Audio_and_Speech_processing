clear all;
close all;

[speech, fs] = audioread('Audio_files/clean_speech.wav');
[noise, ~] = audioread('Audio_files/babble_noise.wav');
soundsc(speech, fs);
pause(length(speech) / fs);

audio = speech + noise(1:length(speech));
soundsc(audio, fs);

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

% Transform to frequency domain
transformed_audio = fft(windowed_audio, window_length, 2);

% Periodogram PSD estimate
psd_est = (abs(transformed_audio) .^ 2) ./ window_length;

noise_psd_est = zeros(size(psd_est, 1), size(psd_est, 2));
for i = 2:size(psd_est, 1)
    noise_psd_est(i, :) = min(psd_est(i - 1, :));
end
% Do noise reduction


% Transform to time domain
recovered_windowed_audio = real(ifft(transformed_audio, window_length, 2));

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

error = sum((audio(1:recovered_length) - recovered_audio).^2);
fprintf('Sum of squared errors: %.4f\n', error);
% Plot to compare visually
figure();
subplot(2,1,1);
plot(audio);
title('Original audio signal');
subplot(2,1,2);
plot(recovered_audio);
title('Reconstructed audio signal');
