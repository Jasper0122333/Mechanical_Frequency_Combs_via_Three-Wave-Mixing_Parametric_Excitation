%% tcmt_fit_v18.m
% TCMT fit for PMUT frequency response.
%
% CHANGES FROM v17 (two bracketing modes for Fano notch at 20 kHz):
%   - manual_modes_kHz: [19.0] -> [19.0, 21.0]
%   - manual_modes_FWHM_Hz: [500] -> [500, 500]
%
%   v17's single 19 kHz mode failed to deepen the 20 kHz dip and instead
%   destabilized fits at 50 kHz and 165-185 kHz. The optimizer found that
%   the extra mode was more useful elsewhere than at 20 kHz, because a
%   single asymmetric flanker cannot produce a deep symmetric notch.
%
%   v18 tests the textbook Fano two-mode configuration: a pair of narrow
%   modes flanking the dip frequency. Two narrow modes with opposite C
%   phases at f0_dip can produce a sharp transmission zero between them,
%   analogous to coupled-resonator transmission filters. This is the only
%   parameter geometry that yields |T| -> 0 at a specific frequency in
%   TCMT.
%
%   If this fails, the model structurally cannot reach 10^-3 depth.
%   NOTHING ELSE CHANGED from v17 (which differed from v16 only in
%   having the injection mechanism).
%
% CHANGES FROM v16 (manual mode injection at 19 kHz for deep 20 kHz dip):
%   - Added manual_modes_kHz / manual_modes_FWHM_Hz parameters.
%     v16 captured 5 of 6 major dips beautifully. The remaining miss at
%     20 kHz is structural: the data has |T|~10^-3 there (3 orders of
%     magnitude below median), which requires near-perfect destructive
%     interference between t and the local mode tails. The detected mode
%     at 20.5 kHz (broad, FWHM~5 kHz, Q~4) alone cannot produce a sharp
%     antiresonance of that depth.
%
%     Manually inject a sharp narrow mode at 19 kHz (FWHM 500 Hz, Q~38)
%     just below the dip. With two modes flanking the dip frequency, the
%     optimizer has the degrees of freedom to set up a Fano-style deep
%     interference notch.
%
%     If the dip is truly inter-modal (two narrow peaks bracketing it),
%     the optimizer will place the new mode and refine kappa/C
%     accordingly. If the dip needs a different structure, the manual
%     mode will adapt or contribute minimally. The 19 kHz seed is the
%     gap between detected modes at 7 kHz and 20.5 kHz.
%     NOTHING ELSE CHANGED from v16.
%
% CHANGES FROM v15 (looser credibility gate for dip phase fitting):
%   - credibility_sigma: 2.0 -> 4.0. v15's credibility weighting was still
%     gating phase residual too aggressively at deep dips: at the 20 kHz
%     dip (log10 mag mismatch ~2.7), only 16% of phase weight was reaching
%     the optimizer, so it had no signal to phase-tune the destructive
%     interference. At sigma=4, that increases to ~63%.
%
%     Phase weight at common dip mismatches:
%       r_m=1 (10x):    sigma=2 -> 78%,  sigma=4 -> 94%
%       r_m=2 (100x):   sigma=2 -> 37%,  sigma=4 -> 78%
%       r_m=2.7 (500x): sigma=2 -> 16%,  sigma=4 -> 63%
%       r_m=3 (1000x):  sigma=2 -> 11%,  sigma=4 -> 57%
%
%     The original sigma=1/2 protection was for the case when global phase
%     drift could fool magnitude-mismatched points. With amp correction and
%     tau in the model, global drift is no longer the issue — local phase
%     residual at dips is now informative and we want it weighted in.
%     NOTHING ELSE CHANGED from v15.
%
% CHANGES FROM v13 (Tikhonov regularization for Stage 1B stability):
%   NOTE: v14 (Q_floor=2) was a regression and is skipped. Starting from v13.
%
%   - Added L2 regularization on C coefficients (regularize_C).
%     The Stage 1B singular Jacobian (optimality jumps to 1e10+ and freezes
%     for 500+ iters) is caused by null-space directions in the parameter
%     space: nearby modes with overlapping kappa can rotate their C
%     coefficients while keeping the sum constant. The Jacobian becomes
%     rank-deficient and TRR cannot make progress.
%
%     A small L2 penalty on Re(C), Im(C) adds diagonal stiffness that
%     breaks these degeneracies — each null-space direction now has a
%     unique penalty minimum (split coupling equally between paired modes).
%     Magnitude chosen to be small enough not to bias the fit but enough
%     to remove the rank-deficiency. NOTHING ELSE CHANGED from v13.
%
% CHANGES FROM v12 (single change to address mode-pegging):
%   - Q_floor: 10 -> 3. v12's run pegged 14 of 36 modes at exactly Q=10,
%     indicating real physical modes with natural Q in 3-15 (substrate
%     spurious modes with acoustic radiation loss) were getting forced
%     upward. Q_floor=3 still rules out v11's Q<1 degeneracy but lets
%     broad physical modes find their natural width. NOTHING ELSE CHANGED.
%
% CHANGES FROM v11 (fix the broad-mode degenerate solution):
%   - Q_FLOOR ADDED: per-mode kappa_ub = w0/Q_floor. v11's kappa_ub was
%     fixed at 2*pi*span*5/w_ref which allowed FWHM > 2 MHz (wider than
%     the entire band). The optimizer used this freedom to create
%     overdamped modes with Q<1 that filled in the baseline instead of
%     using t. v12 caps each mode's width at f0/Q_floor.
%   - TAU CONSTRAINED TO BE CAUSAL: tau_n_min = 0. v11 let tau go
%     negative (-4 us in user's run) which is non-causal and only
%     possible because Q<1 modes provided degenerate compensation.
%   - TIGHTER t_bound: t_bound_mult = 3 (was 5).
%   - PHASE WEIGHT slightly reduced: stage1B = 0.15 (was 0.2).
%   - ITERATION BUDGET REDUCED to 5e4 / 1000 (v11's 1.5e5 hit cap
%     because of bad landscape, not because iters were short).
%
% CHANGES FROM v10 (preserved from v11):
%   - AMPLIFIER CORRECTION ADDED. The AALab A-301 HS amplifier has
%     a 200 kHz bandwidth, so above ~150 kHz the measured response
%     is PMUT_response * H_amp(f) where H_amp(f) ~ 1/(1 + jf/f_pole).
%     v11 divides Tmeas by H_amp before fitting, so the optimizer
%     fits pure PMUT response. Set apply_amp_correction=false to
%     disable (then the model also has to absorb amp rolloff/phase).
%   - SOFTER PHASE CREDIBILITY: v10's credibility = exp(-r_m^2) was
%     too aggressive at deep dips (where r_m=3 means credibility~1e-4,
%     killing the phase signal that's needed to phase-tune destructive
%     interference). v11 uses exp(-(r_m/2)^2): at r_m=3 credibility is
%     ~0.11, so phase still pulls on dips. Tunable via credibility_sigma.
%   - PHASE WEIGHT BUMPED from 0.1 to 0.2 in Stage 1B and elsewhere
%     (only after amp correction is in place; with raw data 0.2 would
%     have been too aggressive).
%
% CHANGES FROM v9 (preserved):
%   - GROUP DELAY tau ADDED TO MODEL. The bare TCMT model
%       T(w) = t + sum C_i/(kappa_i/2 + j(w0_i - w))
%     cannot represent the instrument/cable group delay seen as a
%     linear phase drift in the residual above ~200 kHz. v10 model:
%       T(w) = exp(-j*w*tau) * [t + sum C_i/(...)]
%     Parameter vector now has tau_n at p(3), shifting all kappa/C
%     indices by 1. tau_n is normalized: tau (seconds) = tau_n / w_ref.
%   - INCREASED ITERATION BUDGET: maxFunEvals_global = 1.5e5,
%     maxIter_global = 2000. v9 hit the 4e4 cap because the optimizer
%     was still making progress fighting the missing tau term.
%
% Note: without tau, the optimizer was using wrong-sign imag(C_i) in
% individual modes to partially mimic the global linear phase ramp.
% Once tau absorbs the ramp, those C values should converge to correct
% signs, and the phase rotation direction at each mode should match
% the data.
%
% CHANGES PRESERVED FROM v9:
%   - Two-stage fit (1A magnitude-only, 1B add phase)
%   - Phase credibility weighting
%   - Narrower initial kappa (kappa_init_factor = 0.25)
%   - Smaller local refinement window (local_window_FWHM = 4)
% CHANGES PRESERVED FROM v8:
%   - f_detect_min = 1 kHz, max_modes = 30
%   - f_focus = 100 kHz, focus_boost = 2.0
% CHANGES PRESERVED FROM v7:
%   - Tighter t_bound, Q_ceiling, residual peeling
%
% PARAMETERS TO TUNE FIRST IF FIT IS POOR (in order):
%   1. f_detect_min       — raise if detection picks up noise spikes
%   2. max_modes          — raise if mode count exceeds detection
%   3. Q_ceiling          — lower if modes collapse to narrow spikes
%   4. focus_boost        — raise if low-freq region underfit
%   5. phase_weight       — raise CAUTIOUSLY (max 0.3) for cleaner phase
%
% PARAMETERS TO LEAVE ALONE UNLESS YOU KNOW WHY:
%   - lowT_power = 0.5    (original v6, raising it broke things)
%   - weight_gain = 200   (original v6 — strongly peak-centric is OK
%                           when balanced with lowT_power)

clear; close all; clc;

%% ===================== USER SETTINGS =====================
fname = 'small_signal_Bridge_new.txt';

% --- frequency window ---
f_max_fit       = 400e3;
f_detect_min    = 1e3;      % CHANGED v8: was 25e3 — fundamental at ~3 kHz!
f_focus         = 100e3;    % CHANGED v8: was 150e3 — emphasize operating band
focus_boost     = 2.0;      % CHANGED v8: was 1.5 — stronger low-freq emphasis

% --- detection ---
max_modes       = 30;       % CHANGED v8: was 22 — accommodate ~6 low-freq modes
smooth_pts      = 5;
min_dist_frac   = 1.8e-4;   % CHANGED from 2.5e-4 (modest tighten)
detect_on_log   = true;

% --- manual mode injection (NEW v17) ---
% Seed extra modes the auto-detection misses. Empty arrays disable.
% Use for known features the detector cannot reach because of smoothing
% or because the feature is an inter-modal interference rather than a
% peak/dip per se. Modes are added to f0_list/FWHM_list after detection.
manual_modes_kHz    = [19.0, 21.0];   % v18: bracket 20 kHz dip with two narrow modes
manual_modes_FWHM_Hz = [500, 500];   % v18: both narrow (Q~40), Fano flankers

% --- weights (KEEP v6 ORIGINAL — these work) ---
weight_gain        = 200;   % v6 original
weight_width_mult  = 1.0;   % v6 original
use_lowT_weight    = true;  % v6 original
lowT_floor         = 1e-2;
lowT_power         = 0.5;   % v6 original — DO NOT raise (broke v6 at 1.0)

% --- amp correction (NEW v11) ---
apply_amp_correction = true;   % divide Tmeas by H_amp before fitting
f_amp_pole_Hz        = 200e3;  % AALab A-301 HS bandwidth (-3 dB)
amp_order            = 1;      % single-pole low-pass approximation

% --- objective ---
fit_mode             = 'logmag_phase';
phase_weight_stage1A = 0.0;   % NEW v9: magnitude-only first pass
phase_weight_stage1B = 0.15;  % CHANGED v12: was 0.2 (less aggressive)
phase_weight_final   = 0.15;  % CHANGED v12: was 0.2
use_phase_credibility = true; % NEW v9: down-weight phase where mag is wrong
credibility_sigma    = 4.0;   % CHANGED v16: was 2.0. Phase residual at deep

% --- bounds ---
kappa_min_factor = 0.05;
kappa_init_factor = 0.25;     % NEW v9: shrink detected FWHM for init guess
t_bound_mult     = 3;         % CHANGED v12: was 5 (tighter)
Q_ceiling        = 1500;      % max Q: FWHM >= f0/Q_ceiling
Q_floor          = 3;         % v13 (kept): per-mode kappa upper bound. Q<3
                              % rules out v11's catastrophic degeneracy.
                              % PMUT substrate spurious modes naturally Q=3-15.

% --- regularization (NEW v15) ---
regularize_C     = 0.01;      % L2 penalty weight on each Re/Im(C) coefficient,
                              % normalized by the C bound. Breaks rank-deficient
                              % Jacobian directions from mode-collision pairs.
                              % Range: 0.001-0.05. 0 disables regularization.

% --- group delay (NEW v10) ---
% tau_n is the normalized group delay; actual delay tau (s) = tau_n / w_ref.
% Bound it generously since we don't know the sign convention a priori.
tau_n_init       = 0;          % start with no delay; optimizer will find it
tau_n_min        = 0;          % NEW v12: causal delay only (was -tau_n_max)
tau_n_max        = 30;         % |tau| <= 30/w_ref (~ 24 us at typical w_ref)
% To DISABLE the delay term, set tau_n_max = 0 (forces tau_n = 0).

% --- residual peeling (NEW, optional) ---
do_peel             = true;
n_peel_iters        = 1;     % conservative: 1 pass only
peel_prom_thresh    = 0.15;  % log10 units; modes only added for clear misses
peel_max_new        = 6;     % cap to prevent runaway

% --- optimization ---
maxFunEvals_global = 5e4;     % CHANGED v12: back to ~v10 level (4e4)
maxIter_global     = 1000;    % CHANGED v12: was 2000
maxFunEvals_local  = 5e3;
maxIter_local      = 300;

do_local_refine    = true;
local_window_FWHM  = 4;       % CHANGED v9: was 8 (too wide for broad features)

% --- plotting (Nature compliant) ---
figWidth         = 7.2;     % Nature double column (~183 mm)
figHeight_mag    = 2.6;     % was 2.0 — y-axis spans 5 decades
figHeight_phase  = 2.2;
figHeight_res    = 4.0;
fontSize_axis    = 7;       % was 5 — Nature allows 5-7pt; 7 is far more readable
fontSize_legend  = 6;
fontSize_panel   = 8;       % bold panel label
fontName         = 'Arial';
grid_alpha       = 0.15;    % subtle grid
% =========================================================

%% ===================== LOAD & CROP =====================
raw = readmatrix(fname);
if size(raw,2) < 3, error('Need 3 cols: f_Hz, amplitude, phase(deg).'); end
f = raw(:,1); amp = raw(:,2); phd = raw(:,3);
m = isfinite(f) & isfinite(amp) & isfinite(phd) & (amp >= 0);
f = f(m); amp = amp(m); phd = phd(m);
[f, ord] = sort(f); amp = amp(ord); phd = phd(ord);

N_total = numel(f);
crop_mask = (f <= f_max_fit);
f = f(crop_mask); amp = amp(crop_mask); phd = phd(crop_mask);
fprintf('Cropped: %d -> %d points (f <= %.0f kHz)\n', N_total, numel(f), f_max_fit/1e3);

spanHz = max(f) - min(f);
phd = mod(phd+180, 360) - 180;
Tmeas = (amp./median(amp)) .* exp(1j*deg2rad(phd));

% === v11: amplifier transfer-function correction ===
% A-301 HS has a 200 kHz bandwidth. Above this, measured response
% includes amp rolloff. Divide Tmeas by H_amp to recover pure PMUT response.
if apply_amp_correction
    omega_pole = 2*pi*f_amp_pole_Hz;
    omega_meas = 2*pi*f;
    if amp_order == 1
        H_amp = 1 ./ (1 + 1j*omega_meas/omega_pole);
    elseif amp_order == 2
        H_amp = 1 ./ (1 + 1j*omega_meas/omega_pole).^2;
    else
        error('amp_order must be 1 or 2');
    end
    Tmeas_raw = Tmeas;             % keep raw for plotting comparison
    Tmeas = Tmeas ./ H_amp;        % corrected data used for fitting
    fprintf('Amp correction applied: f_3dB = %.0f kHz, order = %d\n', ...
        f_amp_pole_Hz/1e3, amp_order);
    fprintf('  At f_max=%.0f kHz: |H_amp|=%.3f, arg(H_amp)=%.1f deg\n', ...
        f(end)/1e3, abs(H_amp(end)), rad2deg(angle(H_amp(end))));
else
    Tmeas_raw = Tmeas;
end

w     = 2*pi*f;
w_ref = 2*pi*median(f);
w_n   = w/w_ref;
absT  = max(abs(Tmeas), 1e-12);

fprintf('Fit window [%.1f, %.1f] kHz | |T|: min=%.3g, med=%.3g, max=%.2f\n', ...
        f(1)/1e3, f(end)/1e3, min(absT), median(absT), max(absT));

%% ===================== DETECTION =====================
if detect_on_log
    y_det = log10(max(abs(Tmeas),1e-6));
else
    y_det = abs(Tmeas);
end
y_s = movmean(y_det, smooth_pts);

% Mask: only detect within [f_detect_min, f_max_fit]
detect_mask = (f >= f_detect_min);

min_dist_hz = max(min_dist_frac*spanHz, 3*median(diff(f)));
res_n = y_det - y_s;
sigma = max(1.4826*median(abs(res_n-median(res_n))), 1e-12);
yspan = max(y_s) - min(y_s);

prom_list = [max(3*sigma,0.01*yspan), max(2*sigma,0.005*yspan), ...
             max(1*sigma,0.002*yspan), 0];

f_det = f(detect_mask); y_det_masked = y_s(detect_mask);
[locs_pk,widths_pk,proms_pk] = detect_features(y_det_masked, f_det, prom_list, min_dist_hz, +1);
[locs_dp,widths_dp,proms_dp] = detect_features(y_det_masked, f_det, prom_list, min_dist_hz, -1);
if isempty(locs_pk) && isempty(locs_dp), error('Detection failed.'); end

locs_all   = [locs_pk(:);   locs_dp(:)];
widths_all = [widths_pk(:); widths_dp(:)];
proms_all  = [proms_pk(:);  proms_dp(:)];
type_all   = [repmat("peak",numel(locs_pk),1); repmat("dip",numel(locs_dp),1)];
[locs_all, sidx] = sort(locs_all);
widths_all = widths_all(sidx); proms_all = proms_all(sidx); type_all = type_all(sidx);

% Dedupe close-by features
tol_hz = 0.5*min_dist_hz;
keep = true(size(locs_all));
i = 1;
while i <= numel(locs_all)
    j = i + 1;
    while j <= numel(locs_all) && abs(locs_all(j) - locs_all(i)) < tol_hz
        if proms_all(j) > proms_all(i), keep(i) = false; i = j;
        else, keep(j) = false; end
        j = j + 1;
    end
    i = j;
end
locs_all = locs_all(keep); widths_all = widths_all(keep);
proms_all = proms_all(keep); type_all = type_all(keep);

% Top-N by prominence
[~,idx] = sort(proms_all, 'descend');
idx = idx(1:min(max_modes, numel(idx)));
f0_list = locs_all(idx); FWHM_list = widths_all(idx);
proms_used = proms_all(idx); type_used = type_all(idx);

% Sort by frequency
[f0_list, ord2] = sort(f0_list);
FWHM_list = FWHM_list(ord2); proms_used = proms_used(ord2); type_used = type_used(ord2);

% v17: inject manual modes BEFORE computing nModes
if ~isempty(manual_modes_kHz)
    assert(numel(manual_modes_kHz) == numel(manual_modes_FWHM_Hz), ...
        'manual_modes_kHz and manual_modes_FWHM_Hz must match in length');
    f0_manual = manual_modes_kHz(:) * 1e3;
    FWHM_manual = manual_modes_FWHM_Hz(:);
    % Append with placeholder prominence and type
    f0_list = [f0_list(:); f0_manual];
    FWHM_list = [FWHM_list(:); FWHM_manual];
    proms_used = [proms_used(:); zeros(numel(f0_manual),1)];
    type_used = [type_used(:); repmat("manual", numel(f0_manual), 1)];
    % Re-sort by frequency after injection
    [f0_list, ord_m] = sort(f0_list);
    FWHM_list = FWHM_list(ord_m);
    proms_used = proms_used(ord_m);
    type_used = type_used(ord_m);
    fprintf('Injected %d manual mode(s) at: ', numel(f0_manual));
    fprintf('%.2f ', f0_manual/1e3); fprintf('kHz\n');
end

nModes  = numel(f0_list);
w0_list = 2*pi*f0_list(:).';
w0_n    = w0_list/w_ref;

fprintf('\n=== Detected %d resonances above %.0f kHz (min_dist=%.0f Hz) ===\n', ...
    nModes, f_detect_min/1e3, min_dist_hz);
disp(table((1:nModes)', f0_list(:)/1e3, FWHM_list(:), proms_used(:), type_used(:), ...
    'VariableNames',{'idx','f0_kHz','FWHM_Hz','strength','type'}));

%% ===================== INITIAL GUESSES =====================
% NOTE v9: kappa_init_factor (default 0.25) shrinks the detected FWHM
% before using it as initial kappa. Detection smooths the data and
% reports inflated linewidths; starting narrower lets the optimizer
% reach deep antiresonance dips that need sharp modes.
FWHM_safe = max(FWHM_list(:).', spanHz/1e6);
kappa0_n  = 2*pi*(FWHM_safe * kappa_init_factor) / w_ref;

n_bg = max(1, round(0.05*numel(f)));
t0   = mean(Tmeas(end-n_bg+1:end));
t_re0 = real(t0); t_im0 = imag(t0);

idx_f0 = arrayfun(@(fk) find_nearest_idx(f,fk), f0_list);
T_at_peaks = Tmeas(idx_f0);
Ci0_n = (T_at_peaks(:).' - t0) .* (kappa0_n/2);

% v10: p0 = [t_re, t_im, tau_n, kappa_n, Re(C), Im(C)]
p0 = [t_re0, t_im0, tau_n_init, kappa0_n, real(Ci0_n), imag(Ci0_n)];
nP = numel(p0);
fprintf('Initial: |t0|=%.3f (target ~%.2f = median |T|), tau_n=%.3f\n', ...
    abs(t0), median(absT), tau_n_init);

%% ===================== BOUNDS =====================
[lb, ub] = build_bounds(f0_list, FWHM_list, absT, spanHz, w_ref, ...
                       t_bound_mult, Q_ceiling, Q_floor, ...
                       kappa_min_factor, tau_n_min, tau_n_max);
fprintf('Bounds: t_bound=%.3f, Q in [%d, %d], tau_n in [%.1f, %.1f]\n', ...
    t_bound_mult*median(absT), Q_floor, Q_ceiling, tau_n_min, tau_n_max);

% v15: characteristic C scale for Tikhonov regularization, derived from
% the C bound used in build_bounds. Each |C_i|/C_scale residual is O(1)
% at the bound, so regularize_C directly sets the penalty magnitude.
C_scale = max(ub(4+nModes : 3+2*nModes));
fprintf('Regularization: regularize_C=%.3g, C_scale=%.3g\n', regularize_C, C_scale);
p0 = min(max(p0, lb), ub);  % clip to bounds

%% ===================== WEIGHTS =====================
sqrtW = build_weights(f, f0_list, FWHM_list, absT, spanHz, ...
                     weight_gain, weight_width_mult, use_lowT_weight, ...
                     lowT_floor, lowT_power, f_focus, focus_boost);

%% ===================== STAGE 1A: GLOBAL FIT (MAGNITUDE-ONLY) =====================
% Get the magnitude basin of attraction without phase residual saturation
% blocking access to deep dips.
opts1 = optimoptions('lsqnonlin', ...
    'Display','iter-detailed', ...
    'MaxFunctionEvaluations', maxFunEvals_global, ...
    'MaxIterations', maxIter_global, ...
    'FunctionTolerance', 1e-10, ...
    'StepTolerance', 1e-11);

obj_1A = @(p) build_residual(p, w_n, w0_n, nModes, Tmeas, sqrtW, ...
                             fit_mode, phase_weight_stage1A, use_phase_credibility, ...
                             credibility_sigma, regularize_C, C_scale);

fprintf('\n=== Stage 1A: magnitude-only, %d modes, %d params ===\n', nModes, nP);
tic;
[pFit, resnorm, ~, exitflag] = lsqnonlin(obj_1A, p0, lb, ub, opts1);
fprintf('Stage 1A: %.1f s, exitflag=%d, resnorm=%.4g, |t|=%.3f\n', ...
    toc, exitflag, resnorm, sqrt(pFit(1)^2 + pFit(2)^2));

%% ===================== STAGE 1B: GLOBAL FIT (ADD PHASE) =====================
% With magnitude in the right basin, add phase at low weight to refine.
% Phase credibility weighting (in build_residual) ensures phase residual
% only contributes where magnitudes already agree.
obj_1B = @(p) build_residual(p, w_n, w0_n, nModes, Tmeas, sqrtW, ...
                             fit_mode, phase_weight_stage1B, use_phase_credibility, ...
                             credibility_sigma, regularize_C, C_scale);

fprintf('\n=== Stage 1B: add phase (weight=%.2f, credibility=%d) ===\n', ...
    phase_weight_stage1B, use_phase_credibility);
tic;
[pFit, resnorm, ~, exitflag] = lsqnonlin(obj_1B, pFit, lb, ub, opts1);
fprintf('Stage 1B: %.1f s, exitflag=%d, resnorm=%.4g, |t|=%.3f\n', ...
    toc, exitflag, resnorm, sqrt(pFit(1)^2 + pFit(2)^2));

%% ===================== STAGE 1B: RESIDUAL PEELING (OPTIONAL) =====================
if do_peel
    for peel_iter = 1:n_peel_iters
        fprintf('\n=== Peel iter %d: searching for missed features ===\n', peel_iter);
        Tfit_cur = tcmt_T(pFit, w_n, w0_n, nModes);
        eps0 = 1e-12;
        r_log = log10(max(absT,eps0)) - log10(max(abs(Tfit_cur),eps0));
        r_smooth = movmean(abs(r_log), smooth_pts);

        % Detect in residual, restricted to f >= f_detect_min
        r_for_detect = r_smooth;
        r_for_detect(f < f_detect_min) = 0;
        [~, new_locs, new_widths, new_proms] = findpeaks(r_for_detect, f, ...
            'MinPeakProminence', peel_prom_thresh, ...
            'MinPeakDistance', 2*min_dist_hz, ...
            'WidthReference', 'halfheight');

        % Exclude any near existing modes
        keep_new = true(size(new_locs));
        for k = 1:numel(new_locs)
            if min(abs(new_locs(k) - f0_list)) < 2*min_dist_hz
                keep_new(k) = false;
            end
        end
        new_locs = new_locs(keep_new);
        new_widths = new_widths(keep_new);
        new_proms = new_proms(keep_new);

        % Cap how many we add per iteration
        if numel(new_locs) > peel_max_new
            [~, idx_top] = sort(new_proms, 'descend');
            idx_top = idx_top(1:peel_max_new);
            new_locs = new_locs(idx_top);
            new_widths = new_widths(idx_top);
        end

        if isempty(new_locs)
            fprintf('  No significant residual features found. Done peeling.\n');
            break;
        end

        nNew = numel(new_locs);
        fprintf('  Adding %d new modes at: %s kHz\n', nNew, ...
            mat2str(round(new_locs(:)'/1e3, 1)));

        % Initial guesses for new modes
        kappa_new_n = 2*pi*new_widths(:).' / w_ref;
        C_new_n = zeros(1, nNew);
        for k = 1:nNew
            iN = find_nearest_idx(f, new_locs(k));
            C_new_n(k) = (Tmeas(iN) - Tfit_cur(iN)) * (kappa_new_n(k)/2);
        end

        % Extend parameter vector (insert new modes, then sort all by f0)
        % v10: indices shifted by 1 due to tau_n at p(3)
        t_re = pFit(1); t_im = pFit(2); tau_n = pFit(3);
        kappa_old = pFit(4 : 3+nModes);
        ReC_old   = pFit(4+nModes : 3+2*nModes);
        ImC_old   = pFit(4+2*nModes : 3+3*nModes);

        f0_combined   = [f0_list(:); new_locs(:)];
        FWHM_combined = [FWHM_list(:); new_widths(:)];
        kappa_combined = [kappa_old(:); kappa_new_n(:)];
        ReC_combined   = [ReC_old(:); real(C_new_n(:))];
        ImC_combined   = [ImC_old(:); imag(C_new_n(:))];

        [f0_list, so] = sort(f0_combined);
        FWHM_list = FWHM_combined(so);
        kappa_combined = kappa_combined(so);
        ReC_combined = ReC_combined(so);
        ImC_combined = ImC_combined(so);

        nModes = numel(f0_list);
        w0_list = 2*pi*f0_list(:).';
        w0_n = w0_list/w_ref;

        pFit = [t_re, t_im, tau_n, kappa_combined(:).', ReC_combined(:).', ImC_combined(:).'];
        nP = numel(pFit);

        % Rebuild bounds and weights for new mode set
        [lb, ub] = build_bounds(f0_list, FWHM_list, absT, spanHz, w_ref, ...
                               t_bound_mult, Q_ceiling, Q_floor, ...
                               kappa_min_factor, tau_n_min, tau_n_max);
        sqrtW = build_weights(f, f0_list, FWHM_list, absT, spanHz, ...
                             weight_gain, weight_width_mult, use_lowT_weight, ...
                             lowT_floor, lowT_power, f_focus, focus_boost);
        pFit = min(max(pFit, lb), ub);

        % Refit
        obj = @(p) build_residual(p, w_n, w0_n, nModes, Tmeas, sqrtW, ...
                                  fit_mode, phase_weight_final, use_phase_credibility, ...
                                  credibility_sigma, regularize_C, C_scale);
        opts_peel = optimoptions('lsqnonlin', 'Display','off', ...
            'MaxFunctionEvaluations', maxFunEvals_global, ...
            'MaxIterations', maxIter_global, ...
            'FunctionTolerance', 1e-10, 'StepTolerance', 1e-11);
        tic;
        [pFit, resnorm, ~, exitflag] = lsqnonlin(obj, pFit, lb, ub, opts_peel);
        fprintf('  Refit: %.1f s, exitflag=%d, resnorm=%.4g, nModes=%d, |t|=%.3f\n', ...
            toc, exitflag, resnorm, nModes, sqrt(pFit(1)^2+pFit(2)^2));
    end
end

%% ===================== STAGE 2: LOCAL REFINEMENT =====================
if do_local_refine
    fprintf('\n=== Stage 2: per-mode local refinement ===\n');
    opts2 = optimoptions('lsqnonlin','Display','off', ...
        'MaxFunctionEvaluations', maxFunEvals_local, ...
        'MaxIterations', maxIter_local);

    for i = 1:nModes
        flo = f0_list(i) - local_window_FWHM*FWHM_list(i);
        fhi = f0_list(i) + local_window_FWHM*FWHM_list(i);
        mask = (f >= flo) & (f <= fhi);
        if nnz(mask) < 20, continue; end

        w_n_loc = w_n(mask); Tmeas_loc = Tmeas(mask); sqrtW_loc = sqrtW(mask);
        T_oth = compute_T_others(pFit, w_n_loc, w0_n, nModes, i);

        % v10: indices shifted by 1 due to tau_n at p(3). tau is held
        % fixed at its current global value during per-mode local refit.
        tau_n_cur = pFit(3);
        p_loc0 = [pFit(1), pFit(2), pFit(3+i), pFit(3+nModes+i), pFit(3+2*nModes+i)];
        lb_loc = [lb(1), lb(2), lb(3+i), lb(3+nModes+i), lb(3+2*nModes+i)];
        ub_loc = [ub(1), ub(2), ub(3+i), ub(3+nModes+i), ub(3+2*nModes+i)];

        obj_loc = @(pl) local_residual(pl, w_n_loc, w0_n(i), Tmeas_loc, ...
                                       sqrtW_loc, T_oth, tau_n_cur, fit_mode, ...
                                       phase_weight_final, use_phase_credibility, credibility_sigma);
        try
            p_loc = lsqnonlin(obj_loc, p_loc0, lb_loc, ub_loc, opts2);
            pFit(3+i)          = p_loc(3);
            pFit(3+nModes+i)   = p_loc(4);
            pFit(3+2*nModes+i) = p_loc(5);
        catch ME
            fprintf('  mode %2d: %s\n', i, ME.message);
        end
    end
    fprintf('Stage 2 done. Final |t|=%.3f\n', sqrt(pFit(1)^2+pFit(2)^2));
end

%% ===================== EVALUATE & REPORT =====================
Tfit      = tcmt_T(pFit, w_n, w0_n, nModes);
t_fit     = pFit(1) + 1j*pFit(2);
tau_n_fit = pFit(3);
tau_fit_s = tau_n_fit / w_ref;             % delay in seconds
kappa_fit = pFit(4:3+nModes)*w_ref;
C_fit     = (pFit(4+nModes:3+2*nModes) + 1j*pFit(4+2*nModes:3+3*nModes))*w_ref;
Q_fit     = w0_list ./ max(kappa_fit, 1e-30);

eps0 = 1e-12;
r_log_final = log10(max(absT,eps0)) - log10(max(abs(Tfit),eps0));
r_ph_final  = angle(Tmeas ./ Tfit);

fprintf('\n===== FIT RESULTS =====\n');
fprintf('t = %.4g %+.4gj  (|t|=%.4f, target ~%.3f = median |T|)\n', ...
    real(t_fit), imag(t_fit), abs(t_fit), median(absT));
fprintf('tau (group delay) = %.4g s = %.3f us (tau_n = %.3f)\n', ...
    tau_fit_s, tau_fit_s*1e6, tau_n_fit);
fprintf('log10|T| residual: RMS=%.4f, max|r|=%.3f\n', ...
    sqrt(mean(r_log_final.^2)), max(abs(r_log_final)));
fprintf('phase residual:    RMS=%.4f rad\n', sqrt(mean(r_ph_final.^2)));

Tout = table((1:nModes)', f0_list(:)/1e3, kappa_fit(:)/(2*pi), abs(C_fit(:)), ...
             rad2deg(angle(C_fit(:))), Q_fit(:), ...
    'VariableNames', {'idx','f0_kHz','kappa_Hz','abs_C','arg_C_deg','Q'});
disp(Tout);

%% ===================== PLOTS (Nature compliant) =====================
close all;

% --- Figure 1: Magnitude ---
hMag = figure('Name', 'Magnitude Fit', 'Units', 'inches', ...
    'Position', [1, 1, figWidth, figHeight_mag], 'Color', 'w');

semilogy(f/1e3, max(absT, 1e-12), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8); hold on;
semilogy(f/1e3, max(abs(Tfit), 1e-12), 'r', 'LineWidth', 1.0);
set(gca, 'FontName', fontName, 'FontSize', fontSize_axis, 'LineWidth', 0.6, ...
    'TickDir', 'in', 'Box', 'on', 'XMinorTick', 'on', 'YMinorTick', 'on', ...
    'GridAlpha', grid_alpha, 'MinorGridAlpha', 0.05);
grid on;
xlabel('Frequency (kHz)', 'FontSize', fontSize_axis);
ylabel('Normalized response', 'FontSize', fontSize_axis);
legend({'Measured','TCMT fit'}, 'FontSize', fontSize_legend, ...
    'Location', 'southwest', 'EdgeColor', 'none', 'Color', 'none');
text(-0.06, 1.05, 'a', 'Units', 'normalized', 'FontSize', fontSize_panel, ...
    'FontWeight', 'bold', 'FontName', fontName, 'Parent', gca);
ylim([1e-4 1e2]);  % keep range explicit

% --- Figure 2: Phase ---
hPhase = figure('Name', 'Phase Fit', 'Units', 'inches', ...
    'Position', [1.2, 0.8, figWidth, figHeight_phase], 'Color', 'w');

plot(f/1e3, angle(Tmeas), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8); hold on;
plot(f/1e3, angle(Tfit), 'b', 'LineWidth', 1.0);
set(gca, 'FontName', fontName, 'FontSize', fontSize_axis, 'LineWidth', 0.6, ...
    'TickDir', 'in', 'Box', 'on', 'XMinorTick', 'on', ...
    'GridAlpha', grid_alpha);
grid on;
xlabel('Frequency (kHz)', 'FontSize', fontSize_axis);
ylabel('Phase (rad)', 'FontSize', fontSize_axis);
legend({'Measured','Model'}, 'FontSize', fontSize_legend, ...
    'Location', 'best', 'EdgeColor', 'none', 'Color', 'none');
text(-0.06, 1.05, 'b', 'Units', 'normalized', 'FontSize', fontSize_panel, ...
    'FontWeight', 'bold', 'FontName', fontName, 'Parent', gca);
ylim([-pi pi]);

% --- Figure 3: Residuals ---
hRes = figure('Name', 'Residuals', 'Units', 'inches', ...
    'Position', [1.4, 0.6, figWidth, figHeight_res], 'Color', 'w');

tl = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(f/1e3, r_log_final, '.', 'MarkerSize', 4, 'Color', [0.2 0.2 0.2]);
yline(0, '--', 'Color', [.5 .5 .5], 'LineWidth', 0.8);
set(gca, 'FontName', fontName, 'FontSize', fontSize_axis, ...
    'TickDir', 'in', 'GridAlpha', grid_alpha, 'Box', 'on');
grid on;
ylabel('\Delta log_{10}|T|', 'FontSize', fontSize_axis);
text(-0.06, 1.05, 'c', 'Units', 'normalized', 'FontSize', fontSize_panel, ...
    'FontWeight', 'bold', 'FontName', fontName, 'Parent', gca);

nexttile;
plot(f/1e3, r_ph_final, '.', 'MarkerSize', 4, 'Color', [0.2 0.2 0.2]);
yline(0, '--', 'Color', [.5 .5 .5], 'LineWidth', 0.8);
set(gca, 'FontName', fontName, 'FontSize', fontSize_axis, ...
    'TickDir', 'in', 'GridAlpha', grid_alpha, 'Box', 'on');
grid on;
xlabel('Frequency (kHz)', 'FontSize', fontSize_axis);
ylabel('\Delta Phase (rad)', 'FontSize', fontSize_axis);
ylim([-pi pi]);

%% ===================== EXPORT =====================
exportgraphics(hMag,   'Figure_Mag_Nature.pdf',   'ContentType', 'vector');
exportgraphics(hPhase, 'Figure_Phase_Nature.pdf', 'ContentType', 'vector');
exportgraphics(hRes,   'Figure_Residuals.pdf',    'ContentType', 'vector');
fprintf('\nFigures exported as PDF (vector).\n');

%% ===================== LOCAL FUNCTIONS =====================

function T = tcmt_T(p, w_n, w0_n, nModes)
    % v10: parameter vector is [t_re, t_im, tau_n, kappa_n(1:nM), Re(C)(1:nM), Im(C)(1:nM)]
    t       = p(1) + 1j*p(2);
    tau_n   = p(3);
    kappa_n = max(p(4:3+nModes), 1e-30);
    C       = p(4+nModes:3+2*nModes) + 1j*p(4+2*nModes:3+3*nModes);
    denom   = kappa_n/2 + 1j*(w0_n - w_n(:));
    delay   = exp(-1j*w_n(:)*tau_n);
    T       = delay .* (t + sum(C ./ denom, 2));
end

function res = build_residual(p, w_n, w0_n, nModes, Tmeas, sqrtW, fit_mode, phase_weight, use_credibility, credibility_sigma, regularize_C, C_scale)
    % v15: added regularize_C, C_scale for L2 Tikhonov penalty on C coeffs.
    if nargin < 9,  use_credibility   = false; end
    if nargin < 10, credibility_sigma = 2.0;   end
    if nargin < 11, regularize_C      = 0;     end
    if nargin < 12, C_scale           = 1;     end
    Tmodel = tcmt_T(p, w_n, w0_n, nModes);
    eps0   = 1e-12;
    switch fit_mode
        case 'logmag_only'
            res = sqrtW .* (log10(max(abs(Tmodel),eps0)) - log10(max(abs(Tmeas),eps0)));
        case 'logmag_phase'
            raw_r_m = log10(max(abs(Tmodel),eps0)) - log10(max(abs(Tmeas),eps0));
            r_m = sqrtW .* raw_r_m;
            if phase_weight > 0
                if use_credibility
                    % v11: credibility = exp(-(r_m / sigma)^2)
                    % sigma=1 (v10): r_m=1->0.37, r_m=2->0.018, r_m=3->1e-4 (too sharp)
                    % sigma=2 (v11): r_m=1->0.78, r_m=2->0.37, r_m=3->0.11 (kept)
                    credibility = exp(-(raw_r_m/credibility_sigma).^2);
                else
                    credibility = 1;
                end
                r_p = phase_weight * sqrtW .* credibility .* angle(Tmodel ./ Tmeas);
                res = [r_m; r_p];
            else
                res = r_m;
            end
        otherwise
            error('Unknown fit_mode: %s', fit_mode);
    end

    % v15: Tikhonov regularization on C coefficients.
    % Appends 2*nModes small residuals to break rank-deficient directions.
    if regularize_C > 0 && C_scale > 0
        C_re = p(4+nModes : 3+2*nModes);
        C_im = p(4+2*nModes : 3+3*nModes);
        r_reg = regularize_C * [C_re(:); C_im(:)] / C_scale;
        res = [res; r_reg];
    end
end

function T_oth = compute_T_others(p, w_n, w0_n, nModes, i_skip)
    % v10: indices shifted by 1 due to tau_n at p(3).
    % Returns sum_{j != i_skip} C_j/(kappa_j/2 + j(w0_j - w)) WITHOUT delay or t.
    % Local refinement re-applies delay and t.
    kappa_n = max(p(4:3+nModes), 1e-30);
    C       = p(4+nModes:3+2*nModes) + 1j*p(4+2*nModes:3+3*nModes);
    keep    = true(1, nModes); keep(i_skip) = false;
    denom   = kappa_n(keep)/2 + 1j*(w0_n(keep) - w_n(:));
    T_oth   = sum(C(keep) ./ denom, 2);
end

function res = local_residual(pl, w_n_loc, w0n_i, Tmeas_loc, sqrtW_loc, T_oth, tau_n, fit_mode, phase_weight, use_credibility, credibility_sigma)
    % v10: tau_n is passed in (held fixed during local refinement) and applied
    % as delay factor over (t + T_others + mode_i contribution).
    % v11: credibility_sigma added (default 2.0).
    if nargin < 10, use_credibility   = false; end
    if nargin < 11, credibility_sigma = 2.0;   end
    t_loc  = pl(1) + 1j*pl(2);
    kn_i   = max(pl(3), 1e-30);
    C_i    = pl(4) + 1j*pl(5);
    delay  = exp(-1j*w_n_loc(:)*tau_n);
    Tmodel = delay .* (t_loc + T_oth + C_i ./ (kn_i/2 + 1j*(w0n_i - w_n_loc(:))));
    eps0   = 1e-12;
    switch fit_mode
        case 'logmag_only'
            res = sqrtW_loc .* (log10(max(abs(Tmodel),eps0)) - log10(max(abs(Tmeas_loc),eps0)));
        case 'logmag_phase'
            raw_r_m = log10(max(abs(Tmodel),eps0)) - log10(max(abs(Tmeas_loc),eps0));
            r_m = sqrtW_loc .* raw_r_m;
            if phase_weight > 0
                if use_credibility
                    credibility = exp(-(raw_r_m/credibility_sigma).^2);
                else
                    credibility = 1;
                end
                r_p = phase_weight * sqrtW_loc .* credibility .* angle(Tmodel ./ Tmeas_loc);
                res = [r_m; r_p];
            else
                res = r_m;
            end
    end
end

function [locs, widths, proms] = detect_features(y, f, prom_list, min_dist_hz, sgn)
    locs = []; widths = []; proms = [];
    for prom = prom_list
        [~,l,wid,p] = findpeaks(sgn*y, f, 'MinPeakProminence', prom, ...
            'MinPeakDistance', min_dist_hz, 'WidthReference', 'halfheight');
        if ~isempty(l), locs = l; widths = wid; proms = p; return; end
    end
end

function k = find_nearest_idx(f, f0)
    [~,k] = min(abs(f-f0));
end

function [lb, ub] = build_bounds(f0_list, FWHM_list, absT, spanHz, w_ref, ...
                                t_bound_mult, Q_ceiling, Q_floor, ...
                                kappa_min_factor, tau_n_min, tau_n_max)
    % v12: parameter layout = [t_re, t_im, tau_n, kappa_n(1:nM), Re(C), Im(C)]
    %      Q_floor sets per-mode kappa upper bound (prevents Q<1 degeneracy)
    %      tau_n_min enables causal-only delay (>=0)
    nModes = numel(f0_list);
    nP = 3 + 3*nModes;
    lb = -inf(1, nP); ub = inf(1, nP);

    % t bound: anchored to median |T|, NOT max
    t_bound = t_bound_mult * median(absT);
    lb(1:2) = -t_bound; ub(1:2) = t_bound;

    % tau bound (normalized). Set tau_n_min=0 for causal-only.
    lb(3) = tau_n_min; ub(3) = tau_n_max;

    % kappa bounds. Per-mode upper bound from Q_floor: FWHM <= f0/Q_floor.
    FWHM_floor   = max([FWHM_list(:).' * kappa_min_factor; ...
                        f0_list(:).' / Q_ceiling], [], 1);
    FWHM_ceiling = f0_list(:).' / Q_floor;          % v12: per-mode kappa ub
    kappa_lb_n = 2*pi*max(FWHM_floor,   spanHz/2e6) / w_ref;
    kappa_ub_n = 2*pi*min(FWHM_ceiling, spanHz*5)   / w_ref;
    lb(4:3+nModes) = kappa_lb_n;
    ub(4:3+nModes) = kappa_ub_n;

    % C bounds — proportional to max |T| and max kappa
    C_ub_n = max(absT) * max(kappa_ub_n) * 2;
    lb(4+nModes:3+2*nModes)   = -C_ub_n; ub(4+nModes:3+2*nModes)   = C_ub_n;
    lb(4+2*nModes:3+3*nModes) = -C_ub_n; ub(4+2*nModes:3+3*nModes) = C_ub_n;
end

function sqrtW = build_weights(f, f0_list, FWHM_list, absT, spanHz, ...
                              weight_gain, weight_width_mult, use_lowT_weight, ...
                              lowT_floor, lowT_power, f_focus, focus_boost)
    nModes = numel(f0_list);
    wgt = ones(size(f));
    for i = 1:nModes
        sig_hz = max((FWHM_list(i)/2) * weight_width_mult, spanHz/1e6);
        wgt = wgt + weight_gain * exp(-0.5*((f - f0_list(i))/sig_hz).^2);
    end
    if use_lowT_weight
        wgt = wgt .* ((max(absT, lowT_floor)/lowT_floor).^(-lowT_power));
    end
    wgt((f < f_focus)) = wgt((f < f_focus)) * focus_boost;
    sqrtW = sqrt(wgt(:));
end
