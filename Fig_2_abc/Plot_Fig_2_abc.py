import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def calculate_asd_db(time_data, dt):
    """
    Applies a Hamming window and calculates the Amplitude Spectral Density (ASD) in dB.
    Includes bin size correction and proper scaling for a single-sided spectrum.
    """
    v = np.squeeze(time_data)
    N = len(v)
    fs = 1.0 / dt
    
    # Apply Hamming window
    window = np.hamming(N)
    v_win = v * window
    
    # Real FFT
    V_f = np.fft.rfft(v_win)
    freqs = np.fft.rfftfreq(N, d=dt)
    
    # Power Spectral Density (PSD) calculation
    # Factor 2.0 accounts for positive/negative frequencies (one-sided)
    # Divided by (fs * sum(window**2)) for window weight and bin size correction
    S_xx = (2.0 / (fs * np.sum(window**2))) * (np.abs(V_f)**2)
    
    # DC and Nyquist components shouldn't be doubled
    S_xx[0] /= 2.0
    if N % 2 == 0:
        S_xx[-1] /= 2.0
        
    # Amplitude Spectral Density (ASD)
    asd = np.sqrt(S_xx)
    
    # Convert to dB(mm/s/rootHz), adding a small epsilon to prevent log10(0)
    asd_db = 20 * np.log10(asd + 1e-15)
    
    # Return frequencies in kHz and spectrum
    return freqs / 1000.0, asd_db

def get_simulation_data(dt_target, target_points, sim_filename="model_simulation_data.npz"):
    """
    Loads cached simulation data if available. Otherwise, simulates the non-linear 
    ODE system, matching the timestep and sequence length of the experimental data, 
    and saves the results to an .npz file.
    """
    if os.path.exists(sim_filename):
        print(f"Loading cached simulation data from: {sim_filename}")
        data = np.load(sim_filename)
        # Check if the cache contains the new individual modes
        if 'x1' in data and 'x2' in data:
            return data['t'], data['x1'], data['x2'], data['x_sum']
        else:
            print("Old cache format detected without individual modes. Rerunning simulation...")
        
    print("Running ODE simulation... This may take several minutes.")
    print("The result will be saved to prevent rerunning on future executions.")
    
    # --- Updated Altered Parameters ---
    f1 = 3210.4
    f2 = 94918.8
    gamma1 = 70.57
    gamma2 = 8127.39
    
    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2
    w1_sq = w1**2
    w2_sq = w2**2
    
    # Remaining parameters based on your original MATLAB logic
    alpha2 = 2.183e15 
    alpha_cross2 = alpha2 * 0.001
    lambda_val = 1e9
    Omega = w1 + w2 
    
    # Calculate simulation duration to ensure we have enough steady-state points
    # We add 0.1s to allow initial transients to settle before slicing our target window
    steady_state_duration = target_points * dt_target
    sim_duration = steady_state_duration + 0.5
    
    t_span = (0, sim_duration)
    # Force the solver to evaluate exactly at the target dt
    t_eval = np.arange(0, sim_duration, dt_target)
    
    y0 = [1e-9, 0, 1e-9, 0]
    
    def ode_sys(t, y):
        # y[0]=x1, y[1]=x1_dot, y[2]=x2, y[3]=x2_dot
        return [
            y[1],
            -gamma1*y[1] - w1_sq*y[0] - lambda_val*y[2]*np.cos(Omega*t),
            y[3],
            -gamma2*y[3] - w2_sq*y[2] - alpha2*(y[2]**2) - alpha_cross2*y[2]*y[0] - lambda_val*y[0]*np.cos(Omega*t)
        ]
        
    # Solve Stiff ODE
    sol = solve_ivp(ode_sys, t_span, y0, method='LSODA', t_eval=t_eval, rtol=1e-6, atol=1e-9)
    
    t_full = sol.t
    x1 = sol.y[1] * 0.001 # v1
    x2 = sol.y[3] * 0.001 # v2
    x_sum_full = x1 + x2
    
    # Extract the exact length to match the experimental time domain length perfectly
    if len(t_full) > target_points:
        t_ss = t_full[-target_points:]
        x1_ss = x1[-target_points:]
        x2_ss = x2[-target_points:]
        x_sum_ss = x_sum_full[-target_points:]
    else:
        t_ss = t_full
        x1_ss = x1
        x2_ss = x2
        x_sum_ss = x_sum_full
        
    # Re-zero the time axis for the steady-state slice
    t_ss = t_ss - t_ss[0]
        
    np.savez(sim_filename, t=t_ss, x1=x1_ss, x2=x2_ss, x_sum=x_sum_ss)
    print(f"Simulation saved to: {sim_filename}")
    return t_ss, x1_ss, x2_ss, x_sum_ss

def main():
    # --- Figure and Font Styling Setup ---
    # Explicitly lock ALL text elements to 7pt to prevent relative scaling
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 7,
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })
    
    # --- Load Experimental Data ---
    exp_filename = "TimeData_Pump_98315Hz_DC_0.00V_AC_0.25V.npz"
    if not os.path.exists(exp_filename):
        print(f"Error: {exp_filename} not found in the current directory.")
        return
        
    exp_data = np.load(exp_filename)
    
    # Flatten/squeeze immediately as requested
    t_exp = np.squeeze(exp_data['time_axis'])
    v_exp = np.squeeze(exp_data['time_data'])
    
    # Extract temporal parameters to match domains
    n_points_exp = len(t_exp)
    dt_exp = t_exp[1] - t_exp[0]
    
    # --- Generate/Load Model Data ---
    t_mod, v1_mod, v2_mod, v_sum_mod = get_simulation_data(dt_target=dt_exp, target_points=n_points_exp)
    
    # --- Process ASD ---
    freqs_exp, asd_exp = calculate_asd_db(v_exp, dt_exp)
    freqs_mod_v1, asd_mod_v1 = calculate_asd_db(v1_mod, dt_exp)
    freqs_mod_v2, asd_mod_v2 = calculate_asd_db(v2_mod, dt_exp)
    freqs_mod_sum, asd_mod_sum = calculate_asd_db(v_sum_mod, dt_exp)
    
    # --- Plotting ---
    fig_width = 7.2
    fig_height = fig_width * (2/3) # 4.8 inches
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(fig_width, fig_height), dpi=300)
    
    # Subplot 1: Simulation Spectrum (Individual Modes v1 and v2)
    ax1.plot(freqs_mod_v1, asd_mod_v1, color='red', linewidth=0.5, label='Mode 1 ($v_1$)')
    ax1.plot(freqs_mod_v2, asd_mod_v2, color='green', linewidth=0.5, label='Mode 2 ($v_2$)')
    ax1.set_title('Simulation Spectrum: Velocity of Mode 1 and Mode 2', pad=5)
    ax1.set_ylabel('Amplitude (dB)')
    ax1.set_xlim(0, 200)
    ax1.grid(True, linestyle='--', linewidth=0.3, alpha=0.7)
    ax1.legend(loc='upper right', frameon=False)
    
    # Subplot 2: Model Simulation Spectrum (Sum)
    ax2.plot(freqs_mod_sum, asd_mod_sum, color='black', linewidth=0.5)
    ax2.set_title('Simulation Spectrum: Velocity Sum of Mode 1 + Mode 2', pad=5)
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_xlim(0, 200)
    ax2.grid(True, linestyle='--', linewidth=0.3, alpha=0.7)

    # Subplot 3: Experimental Spectrum
    ax3.plot(freqs_exp, asd_exp, color='blue', linewidth=0.5)
    ax3.set_title('Experimental Spectrum', pad=5)
    ax3.set_xlabel('Frequency (kHz)')
    ax3.set_ylabel('Amplitude (dB)')
    ax3.set_xlim(0, 200)
    ax3.grid(True, linestyle='--', linewidth=0.3, alpha=0.7)
    
    # Add 'a', 'b', and 'c' annotations (Strictly Size 9)
    ax1.text(-0.05, 1.05, 'a', transform=ax1.transAxes, 
             fontsize=9, fontweight='bold', fontname='Arial', va='bottom', ha='left')
    ax2.text(-0.05, 1.05, 'b', transform=ax2.transAxes, 
             fontsize=9, fontweight='bold', fontname='Arial', va='bottom', ha='left')
    ax3.text(-0.05, 1.05, 'c', transform=ax3.transAxes, 
             fontsize=9, fontweight='bold', fontname='Arial', va='bottom', ha='left')
    
    # REPLACED plt.tight_layout() with manual adjustments
    # This prevents the canvas from resizing itself to fit margins
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08, hspace=0.35)
    
    # Save Output
    output_pdf = "Figure_2_abc.pdf"
    
    # REMOVED bbox_inches='tight' to ensure the PDF is exactly 7.2 x 4.8 inches
    plt.savefig(output_pdf, format='pdf')
    print(f"Plot successfully saved to: {output_pdf}")
    # plt.show()

if __name__ == "__main__":
    main()