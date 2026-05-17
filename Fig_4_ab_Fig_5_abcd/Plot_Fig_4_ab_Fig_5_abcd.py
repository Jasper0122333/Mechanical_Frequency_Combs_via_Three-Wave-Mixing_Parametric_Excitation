import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, sosfreqz
from scipy.integrate import solve_ivp

# =========================================================================
# 1. Configuration & Global Aesthetics
# =========================================================================
fileName = '2025_11_21_Bridge_Shacker_106100Hz_AC_0p2_DC_0p6_Shacker_10G_50Hz.csv'
odeFileName = 'ode_sim_results.npz'  # Changed from .mat to native Python .npz

fontName = 'Arial'
baseFontSize = 7
labelFontSize = 9

figWidth = 7.2  # inches
figHeightSpec = figWidth / 3.5
figHeightTime = figWidth / 3.5 

# Configure Matplotlib Aesthetics
plt.rcParams.update({
    'font.family': fontName,
    'font.size': baseFontSize,
    'axes.titlesize': baseFontSize,
    'axes.labelsize': baseFontSize,
    'xtick.labelsize': baseFontSize,
    'ytick.labelsize': baseFontSize,
    'legend.fontsize': baseFontSize,
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
    'font.weight': 'normal',
    'figure.titlesize': baseFontSize,
    'lines.linewidth': 0.75,
    'axes.linewidth': 0.5,           # Thinner plot box
    'xtick.major.width': 0.5,        # Thinner major x-ticks
    'ytick.major.width': 0.5,        # Thinner major y-ticks
    'xtick.minor.width': 0.5,        # Thinner minor x-ticks
    'ytick.minor.width': 0.5,        # Thinner minor y-ticks
    'pdf.fonttype': 42, # Ensures true fonts are embedded in PDF
    'ps.fonttype': 42
})

# --- Axis Limit Controls ---
# Figure 1 (Exp Spectrum)
span_fig1 = 1.0; ylim_fig1 = [-150, 0]
# Figure 2 (Exp Time)
xlim_fig2 = [-60, 60]; xlim_acc_fig2 = [-100, 100]; ylim_fig2 = [0.4, 0.6]
# Figure 3 (Exp ASD)
xlim_fig3 = [0, 0.3]; ylim_fig3 = [-35, 35]
# Figure 4 (Sim Spectrum)
span_fig4 = 1.0; ylim_fig4 = [-150, 0]
# Figure 5 (Sim Time)
xlim_fig5 = [-60, 60]; xlim_acc_fig5 = [-100, 100]; ylim_fig5 = [0.4, 0.6]
# Figure 6 (Sim ASD)
xlim_fig6 = [0, 0.3]; ylim_fig6 = [-35, 35]

# =========================================================================
# 2. Load Experimental Data
# =========================================================================
print(f"Loading experimental data from '{fileName}'...")
try:
    data_table = pd.read_csv(fileName)
    time_exp = data_table.iloc[:, 0].values
    velocity_exp = data_table.iloc[:, 1].values
    
    dt_exp = np.mean(np.diff(time_exp))
    fs_exp = 1 / dt_exp
    L_exp = len(velocity_exp)
except FileNotFoundError:
    print(f"Warning: {fileName} not found. Skipping experimental portion if not present.")

# =========================================================================
# 3. Load or Generate ODE Simulated Data
# =========================================================================
gamma1 = 116.47;     w1_sq = 4.665e8;      w1 = w1_sq**0.5
gamma2 = 11312.9;    w2_sq = 4.183e11;     w2_base = w2_sq**0.5
alpha2 = 2.183e15;   alpha_cross2 = alpha2 * 0.006
lambda_val = 1e9;    Omega = w1 + w2_base

Fs_sim = 1e7
dt_sim = 1 / Fs_sim
t_end = 5
tspan = [0, t_end]
t_eval = np.arange(0, t_end, dt_sim)

if os.path.isfile(odeFileName):
    print(f"Loading simulated model data from '{odeFileName}'...")
    with np.load(odeFileName) as data:
        t_sim_full = data['t_sim_full']
        Y = data['Y']
else:
    print('Simulating ODE Model... This will take a few moments.')
    y0 = [1e-9, 0, 1e-9, 0]
    
    def ode_sys(t, y):
        dy = np.zeros_like(y)
        dy[0] = y[1]
        dy[1] = -gamma1*y[1] - (w1_sq * (1 + 0.0061*np.cos(2*np.pi*50*(t-0.002))))*y[0] - lambda_val*y[2]*np.cos(Omega*t)
        dy[2] = y[3]
        dy[3] = -gamma2*y[3] - w2_sq*y[2] - alpha2*y[2]**2 - alpha_cross2*y[2]*y[0] - lambda_val*y[0]*np.cos(Omega*t)
        return dy

    # BDF is the Python equivalent best suited for stiff systems like MATLAB's ode15s
    sol = solve_ivp(ode_sys, tspan, y0, method='LSODA', t_eval=t_eval, rtol=1e-6, atol=1e-9)
    t_sim_full = sol.t
    Y = sol.y.T
    
    np.savez_compressed(odeFileName, t_sim_full=t_sim_full, Y=Y)
    print('ODE Simulation complete and saved.')

# Extract Steady State Model Data
idx_ss = (t_sim_full >= 1.0)
time_ss = t_sim_full[idx_ss]
time_sim = time_ss - time_ss[0]
v1_ss = Y[idx_ss, 1]
v2_ss = Y[idx_ss, 3]
velocity_sim = 0.1 * v1_ss + v2_ss

L_sim = len(velocity_sim)

# =========================================================================
# 4. Define the 8 Frequency Regions
# =========================================================================
f1_hz_actual_sim = 3464.75
f2_hz_actual_sim = 102908
f_pump_hz_sim = (w1 + w2_base) / (2*np.pi)

regions = [
    {'title': 'Mode 1',          'c_exp': 3491,             'c_sim': f1_hz_actual_sim},
    {'title': '2*(Mode 1)',      'c_exp': 3491 * 2,         'c_sim': 2 * f1_hz_actual_sim},
    {'title': '3*(Mode 1)',      'c_exp': 3491 * 3,         'c_sim': 3 * f1_hz_actual_sim},
    {'title': '4*(Mode 1)',      'c_exp': 3491 * 4,         'c_sim': 4 * f1_hz_actual_sim},
    {'title': 'Mode 2 - Mode 1', 'c_exp': 102608 - 3491,    'c_sim': f2_hz_actual_sim - f1_hz_actual_sim},
    {'title': 'Mode 2',          'c_exp': 102608,           'c_sim': f2_hz_actual_sim},
    {'title': 'Pump + Mode 1',   'c_exp': 106100 + 3491,    'c_sim': f_pump_hz_sim + f1_hz_actual_sim},
    {'title': 'Pump',            'c_exp': 106100,           'c_sim': f_pump_hz_sim}
]

# =========================================================================
# 5. Process Experimental (1) & Model (2)
# =========================================================================
for d_idx in [1, 2]:
    
    if d_idx == 1:
        if 'velocity_exp' not in locals(): continue
        print('--- Generating Figures 1 to 3 (Experimental) ---')
        v_data, fs_data, time_data, L_data = velocity_exp, fs_exp, time_exp, L_exp
        bp_order = 8; lp_fc = 5000; bw = 500
        fig_nums = [1, 2, 3]
        numbers = ['4', '5', '5']
        letters = ['a', 'a', 'c']
        spec_span = span_fig1; y_spec = ylim_fig1
        x_time = xlim_fig2; x_time_acc = xlim_acc_fig2; y_time = ylim_fig2
        x_asd = xlim_fig3; y_asd = ylim_fig3
    else:
        print('--- Generating Figures 4 to 6 (Model) ---')
        v_data, fs_data, time_data, L_data = velocity_sim, Fs_sim, time_sim, L_sim
        bp_order = 8; lp_fc = 5000; bw = 500
        fig_nums = [4, 5, 6]
        numbers = ['4', '5', '5']
        letters = ['b', 'b', 'd']
        spec_span = span_fig4; y_spec = ylim_fig4
        x_time = xlim_fig5; x_time_acc = xlim_acc_fig5; y_time = ylim_fig5
        x_asd = xlim_fig6; y_asd = ylim_fig6
        
    # --- Pre-calculate Original Signal Spectrum (ASD Computation) ---
    win = np.hamming(L_data)
    Y_orig = np.fft.fft(v_data * win)
    
    # Calculate proper ASD scaled by window energy and sampling frequency
    PSD_orig = (2 / (fs_data * np.sum(win**2))) * np.abs(Y_orig[:L_data//2 + 1])**2
    PSD_orig[0] /= 2
    if L_data % 2 == 0: PSD_orig[-1] /= 2
        
    f_vec_hz = fs_data * np.arange(L_data//2 + 1) / L_data
    f_vec_khz = f_vec_hz / 1000
    ASD_orig_dB = 10 * np.log10(PSD_orig + 1e-12)
    
    # --- Prepare Figures & Layouts ---
    fig_spec, axes_spec = plt.subplots(2, 4, figsize=(figWidth, figHeightSpec), constrained_layout=True)
    fig_time, axes_time = plt.subplots(1, 9, figsize=(figWidth, figHeightTime), constrained_layout=True)
    fig_asd, axes_asd = plt.subplots(2, 4, figsize=(figWidth, figHeightSpec), constrained_layout=True)
    
    axes_spec = axes_spec.flatten()
    axes_asd = axes_asd.flatten()
    
    # Add ABCD Annotations strictly at 9pt
    fig_spec.text(0.01, 0.98, letters[0], fontsize=labelFontSize, ha='left', va='top')
    fig_time.text(0.01, 0.98, letters[1], fontsize=labelFontSize, ha='left', va='top')
    fig_asd.text(0.01, 0.98, letters[2], fontsize=labelFontSize, ha='left', va='top')
    
    # Global labels for constrained layout, explicitly forcing 7pt
    fig_spec.supxlabel('Frequency (kHz)', fontsize=baseFontSize)
    fig_spec.supylabel('ASD (dB/\u221AHz)', fontsize=baseFontSize) 
    
    fig_time.supylabel('Time (s)', fontsize=baseFontSize)
    
    fig_asd.supxlabel('Frequency (kHz)', fontsize=baseFontSize)
    fig_asd.supylabel('ASD (dB g/\u221AHz)', fontsize=baseFontSize)
    
    # --- Loop Through 8 Regions ---
    for i in range(8):
        fc = regions[i]['c_exp'] if d_idx == 1 else regions[i]['c_sim']
        fc_khz = fc / 1000
        bw_khz = bw / 1000
        
        # Bandpass Filter
        sos_bp = butter(bp_order, [fc - bw, fc + bw], btype='bandpass', fs=fs_data, output='sos')
        v_filtered = sosfiltfilt(sos_bp, v_data)
        
        # Filtered Spectrum & Response
        Y_filt = np.fft.fft(v_filtered * win)
        PSD_filt = (2 / (fs_data * np.sum(win**2))) * np.abs(Y_filt[:L_data//2 + 1])**2
        PSD_filt[0] /= 2
        if L_data % 2 == 0: PSD_filt[-1] /= 2
        ASD_filt_dB = 10 * np.log10(PSD_filt + 1e-12)
        
        f_resp_hz, h_resp = sosfreqz(sos_bp, worN=8192, fs=fs_data)
        f_resp_khz = f_resp_hz / 1000
        h_resp_dB = 20 * np.log10(np.abs(h_resp) + 1e-12)
        
        # 1. Plot Spectrums
        ax_s = axes_spec[i]
        ax_s.plot(f_vec_khz, ASD_orig_dB, color='0.7', linewidth=0.5, label='Original')
        ax_s.plot(f_vec_khz, ASD_filt_dB, color='b', linewidth=0.75, label='Filtered')
        ax_s.plot(f_resp_khz, h_resp_dB, color='r', linewidth=0.75, linestyle='--', label='Filter Shape')
        
        ax_s.set_ylim(y_spec)
        ax_s.set_xlim([fc_khz - spec_span, fc_khz + spec_span])
        ax_s.set_title(regions[i]['title'], fontsize=baseFontSize)
        ax_s.grid(True, linewidth=0.5)
        
        if i % 4 != 0:
            ax_s.set_yticklabels([])
            ax_s.tick_params(axis='y', left=False, right=False)
            
        # Demodulation
        v_analytic = hilbert(v_filtered)
        inst_phase = np.unwrap(np.angle(v_analytic))
        inst_freq_hz = (fs_data / (2 * np.pi)) * np.gradient(inst_phase)
        freq_dev = inst_freq_hz - fc
        
        # Low-Pass Filter
        sos_lp = butter(6, lp_fc, btype='low', fs=fs_data, output='sos')
        freq_dev_smooth = sosfiltfilt(sos_lp, freq_dev)
        
        # Calculate Max Delta Hz
        time_mask = (time_data >= y_time[0]) & (time_data <= y_time[1])
        if np.any(time_mask):
            max_dev = np.max(np.abs(freq_dev_smooth[time_mask]))
        else:
            max_dev = 0.0
            
        # 2. Plot Time Domain
        ax_t = axes_time[i]
        ax_t.plot(freq_dev_smooth, time_data, 'b', linewidth=0.75)
        ax_t.set_ylim(y_time)
        ax_t.set_xlim(x_time)
        ax_t.set_title(f"{regions[i]['title']}\nMax $\\Delta$: {max_dev:.1f} Hz", fontsize=baseFontSize)
        ax_t.grid(True, linewidth=0.5)
        
        if i != 0:
            ax_t.set_yticklabels([])
            ax_t.tick_params(axis='y', left=False, right=False) 
            
        if i == 3: 
            ax_t.set_xlabel('Frequency Deviation (Hz)', x=1.0, fontsize=baseFontSize)
            
        # ASD Calculation
        Y_demod = np.fft.fft(freq_dev_smooth * win)
        PSD_demod = (2 / (fs_data * np.sum(win**2))) * np.abs(Y_demod[:L_data//2 + 1])**2
        PSD_demod[0] /= 2
        if L_data % 2 == 0: PSD_demod[-1] /= 2
        ASD_dB = 10 * np.log10(PSD_demod + 1e-12)
        
        # 3. Plot Demodulated ASD
        ax_a = axes_asd[i]
        ax_a.plot(f_vec_khz, ASD_dB, 'b', linewidth=0.75) 
        ax_a.set_xlim(x_asd)
        ax_a.set_ylim(y_asd)
        ax_a.set_title(regions[i]['title'], fontsize=baseFontSize)
        ax_a.grid(True, linewidth=0.5)
        
        if i % 4 != 0:
            ax_a.set_yticklabels([])
            ax_a.tick_params(axis='y', left=False, right=False)

    # --- Place Legend natively using layout engine (Removes need for bbox_inches='tight') ---
    handles, labels = axes_spec[0].get_legend_handles_labels()
    fig_spec.legend(handles, labels, loc='outside right center', frameon=False, fontsize=baseFontSize)

    # --- Acceleration Profile (Panel 9 in Time Figure) ---
    ax_acc = axes_time[8]
    
    if d_idx == 1:
        fc_acc = 2000 
        sos_acc = butter(6, fc_acc, btype='low', fs=fs_data, output='sos')
        velocity_lp = sosfiltfilt(sos_acc, v_data)
        acceleration = np.gradient(velocity_lp) * fs_data
    else:
        acceleration = -100 * np.cos(2 * np.pi * 50 * (time_data - 0.002))
        
    ax_acc.plot(acceleration, time_data, 'r', linewidth=0.75)
    ax_acc.set_ylim(y_time)
    ax_acc.set_xlim(x_time_acc)
    ax_acc.set_xlabel('Acceleration (m/s²)', fontsize=baseFontSize)
    ax_acc.set_title('Acceleration', fontsize=baseFontSize)
    ax_acc.grid(True, linewidth=0.5)
    ax_acc.set_yticklabels([])
    ax_acc.tick_params(axis='y', left=False, right=False)
    
    # --- Save as PDFs (REMOVED bbox_inches='tight' to lock 7.2in canvas) ---
    figs = [(fig_spec, letters[0]), (fig_time, letters[1]), (fig_asd, letters[2])]
    
    for fig_obj, letter in figs:
        outName = f"Figure{numbers[figs.index((fig_obj, letter))]}{letter}.pdf"
        fig_obj.savefig(outName, format='pdf', dpi=300)
        print(f"Saved {outName}")
        plt.close(fig_obj)

print('All processing complete and PDFs generated.')