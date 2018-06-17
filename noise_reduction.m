clear all;
close all;

%% Setup
noise_level = 12;
% Read in data
[speech, fs] = audioread('Audio_files/clean_speech.wav');
[noise, ~] = audioread('Audio_files/aritificial_nonstat_noise.wav');

speech = speech(1:115531);
noise = noise_level .* noise(1:length(speech));
audio = speech + noise;

fprintf('Now playing noisy audio\n');
soundsc(audio, fs);
pause(floor(length(audio) / fs) + 1);
% Declare parameters
window_length_sec = 20e-3; %20ms
window_length = window_length_sec * fs;

window = hamming(window_length);
num_windows = 2*floor(length(audio) / window_length) - 1;

alpha_s = .95; % Speech PSD smoothing parameter
alpha_n_max = .96;
alpha_n_min = .3;
M = .890;               % From paper
a_v = 2.12;             % From paper

ms_window_length = 120;  % How many windows to use in noise PSD estimation

%% Windowing
% Generate matrix of windows
windowed_audio = zeros(num_windows, window_length);
step_size = floor(window_length / 2);

window_index = 1;
snr_count = 1;
for i = 1:num_windows
    from = step_size * i - step_size + 1;
    to = step_size * i + step_size;
    windowed_audio(i,:) = window .* audio(from:to);
    curr_snr = snr(speech(from:to), noise(from:to));
    if curr_snr ~= -inf
        snrs(snr_count) = curr_snr;
        snr_count = snr_count + 1;
    end
end

avg_snr = nanmean(snrs);
fprintf('Audio signal has average SNR of %.2fdB\n', avg_snr);
%% Processing
fprintf('Processing windowed audio\n');
% Transform to frequency domain
transformed_audio = fft(windowed_audio, window_length, 2);

% Periodogram PSD estimate
psd_est = abs(transformed_audio) .^ 2;

smoothed_psd = zeros(size(psd_est, 1), size(psd_est, 2));
smoothed_psd(1, :) = psd_est(1,:);
noise_psd_est = psd_est(1,:);

speech_snr_est = zeros(size(psd_est, 1), size(psd_est, 2));
speech_psd_est = zeros(1, size(psd_est, 2));

% Initial alphas
alpha_c = .3;
alpha_n = ones(1,size(smoothed_psd, 2)) .* .96 * alpha_c;

% Parameters for bias estimation
beta = (alpha_n .^2 < .8) .* alpha_n .^2 + (alpha_n .^2 >= .8) .* .8;
psd_bar = (1 - beta) .* smoothed_psd(1, :);
psd_bar_sq = (1 - beta) .* smoothed_psd(1, :).^2;
psd_var_est = psd_bar_sq - psd_bar.^2;
for i = 2:size(psd_est, 1)
    if i == 203
        disp('Hey');
    end
    % Update smoothing parameters
    alpha_c_tilde = 1 /...
        (1 + (sum(smoothed_psd(i - 1,:)) / sum(psd_est(i,:)) - 1))^2;
    alpha_c = 0.7 * alpha_c + 0.3 * max(alpha_c_tilde, 0.7);
    
    alpha_n = (alpha_n_max * alpha_c) ./ ...
        (1 + (smoothed_psd(i - 1, :) ./ noise_psd_est - 1).^2);

    % Clip alpha
    alpha_n(alpha_n > .96) = .96;
    alpha_n(alpha_n < .3) = .3;
    % Smooth the PSD estimate
    smoothed_psd(i, :) = alpha_n .* smoothed_psd(i-1, :) + ...
        (1 - alpha_n) .* psd_est(i, :);

    % Estimate bias
    beta = (alpha_n .^2 < .8) .* alpha_n .^2 + (alpha_n .^2 >= .8) .* .8;
    psd_bar = beta .* psd_bar +...
        (1 - beta) .* smoothed_psd(i, :);
    psd_bar_sq = beta .* psd_bar_sq +...
        (1 - beta) .* smoothed_psd(i, :).^2;
    psd_var_est = psd_bar_sq - psd_bar.^2;
    
    Q_eq = (2 .* noise_psd_est .^ 2) ./ psd_var_est;
    % Clip Q_eq
    Q_eq(Q_eq > 2) = 2;
    Q_tilde_eq = (Q_eq - 2*M) ./ (1 - M);
    noise_bias = 1 + (ms_window_length - 1) .* (2 ./ Q_tilde_eq);
    bias_correction = 1 + (a_v / window_length) *...
        sqrt(sum(1 ./ Q_eq));
    noise_bias = noise_bias .* bias_correction;
    
    
    % Find minimum from last few frames
    if i <= ms_window_length
        noise_psd_est = min(smoothed_psd(1:i, :), [], 1)...
            .* noise_bias;
    else
        noise_psd_est = min(smoothed_psd(i-ms_window_length:i, :),...
            [], 1) .* noise_bias;
    end%if
    speech_snr_est(i, :) =...
        alpha_s .* (speech_psd_est ./ noise_psd_est) +...
        (1 - alpha_s) .* max([smoothed_psd(i, :) ./ noise_psd_est - 1, 0.2]);
    speech_psd_est = speech_snr_est(i, :) .* noise_psd_est;
end%for

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

orig = stoi(speech, audio, fs);
fprintf('STOI score unprocessed: %.4f\n', orig);

intelligibility = stoi(speech(1:recovered_length), recovered_audio, fs);
fprintf('STOI score   processed: %.4f\n', intelligibility);
% Plot to compare visually
figure();
subplot(3,1,1);
plot(speech);
title('Original clean signal');
subplot(3,1,2);
plot(audio);
title('Original noisy signal');
subplot(3,1,3);
plot(recovered_audio);
title('Reconstructed audio signal');
