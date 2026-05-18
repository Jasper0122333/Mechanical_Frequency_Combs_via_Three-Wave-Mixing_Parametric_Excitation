clc;clear;close all;
set(0, 'DefaultLineLineWidth', 1);
set(groot,'defaultAxesFontName','Times New Roman')
set(groot,'defaultAxesFontSize',16)
set(groot,{'DefaultAxesXColor','DefaultAxesYColor','DefaultAxesZColor'},{'k','k','k'});
set(0, 'DefaultAxesLineWidth', 1.5);
disp('Simulating ODE on freq-Vp plane. This might take hours.');


G_N = 51; w0_N = 51;
G_scan = linspace(100,500,G_N)*1e6;
Omega_scan = linspace(96,101,w0_N)*1e3*2*pi;
x1_max = zeros(w0_N,G_N);

figure(5)
clf;
title('Red is comb, blue is no comb');


for n=G_N
    for m=1:w0_N
% --- System Parameters (from experiments)---

% Mode 1
w1 = (98.315-95)*1e3*2*pi;
w1_sq  = w1^2;
gamma1 = 70.57;

% Mode 2 (Representing Mode 6 in your experiment)
w2 = 95*1e3*2*pi;
w2_sq  = w2^2;
gamma2 = 8127.39;
alpha2 = 2.183e15;

% (Initial guesses - tune these to match the comb amplitudes in your data)
alpha_cross2 = alpha2 * 0.001; % Cross-quadratic acting on Mode 2

% Parametric Excitation
% lambda = 1.65e8; % Tune this to overcome threshold damping (gamma1 * gamma2)
lambda = G_scan(n);
% Omega  = w1 + w2;
Omega  = Omega_scan(m);

% --- Simulation Setup ---
Fs = 1e7;           % 1 MHz sampling frequency
dt = 1/Fs;
t_end = 0.5;       % Simulate for 500 milliseconds
tspan = 0:dt:t_end; 

% Initial Conditions: [x1(0), x1_dot(0), x2(0), x2_dot(0)]
ini = 1e-9;
y0 = [ini; 0; ini; 0]; 

% Options for the stiff ODE solver
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-9);

% --- Define the ODE System ---
% y(1) = x1,  y(2) = x1_dot
% y(3) = x2,  y(4) = x2_dot
ode_sys = @(t, y) [
    y(2);
    -gamma1*y(2) - w1_sq*y(1) ...
        - lambda*y(3)*cos(Omega*t);                    % Parametric Pump
        
    y(4);
    -gamma2*y(4) - w2_sq*y(3) ...
        - alpha2*y(3)^2 - alpha_cross2*y(3)*y(1) ...   % Quadratic (Self + Cross)
        - lambda*y(1)*cos(Omega*t)                     % Parametric Pump
];

% --- Solve the ODE ---
[t, Y] = ode15s(ode_sys, tspan, y0, options);

% Extract full solutions for time-domain plotting
x1 = Y(:, 1);
x2 = Y(:, 3);
% x2 = 10*x2;
x_sum = x1 + x2;

% --- Steady State Extraction for FFT ---
% Find indices for the last 0.5 seconds (or the whole signal if t_end < 0.5)
t_start_ss = max(0, t_end - 0.01); 
idx = t >= t_start_ss;

% Extract the steady-state portion of the signals
t_ss = t(idx);
x1_ss = x1(idx);
x2_ss = x2(idx);
x_sum_ss = x_sum(idx);

[fm,vm]=max(x1_ss);
x1_max(m,n)=fm;

figure(4)
clf;
plot(t, x1, 'b', t, x2, 'r', 'LineWidth', 1);hold on;
scatter(t_ss(vm),fm,'ro');
xlabel('Time (s)'); ylabel('Displacement');
legend('x_1', 'x_2'); grid on;
xlim([0 0.5]); xticks([0:0.1:0.5]);

figure(5)
if fm<y0(1)
scatter(Omega/2e3/pi,lambda*1e-6,50,'b.');
else
scatter(Omega/2e3/pi,lambda*1e-6,50,'r.');
end
hold on;
xlim([95 102]); xticks([95:1:102]);
ylim([00 500]);

pause(0.1);
    end
end

save('Sim.mat','G_scan','Omega_scan','x1_max','ini');