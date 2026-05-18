clc;clear;close all;
set(0, 'DefaultLineLineWidth', 1.5);
set(groot,'defaultAxesFontName','Times New Roman')
set(groot,'defaultAxesFontSize',16)
set(groot,{'DefaultAxesXColor','DefaultAxesYColor','DefaultAxesZColor'},{'k','k','k'});
set(0, 'DefaultAxesLineWidth', 1.2);
options = optimset('TolFun',1e-9,'TolX',1e-9,'display','on');

%% Parameters of the modes (from the experiment)
r1 = 70.57*1e-3; w1  = (98.315-95)*2*pi;
r2 = 8127.39*1e-3; w2  = 95*2*pi;
wF0 = -0.5i*r1 + 0.5*sqrt(4*w1^2-r1^2);
w0_scan = linspace(96,110,500)*2*pi;
%% Analalytical expression of the threshold
O2n = @(w0)(wF0-w0).^2+1i*r2*(wF0-w0)-w2^2;
O2p = @(w0)(wF0+w0).^2+1i*r2*(wF0+w0)-w2^2;
wF = @(G,w0) wF0+0.125*G.^2/wF0.*(1./O2n(w0)+1./O2p(w0));
G = @(w0) sqrt(4*r1./imag((1./O2n(w0)+1./O2p(w0))/wF0)); % G = \Gamma

Gth = G(w0_scan); wF_th = wF(Gth,w0_scan); max(abs(imag(wF_th)))

%% Experimental data
load ExpData.mat; V=V*20; %*20 for the amplifier

% extracting the exp threshold line
V0=1.8; Chi_eff0 = 59.5238;
G2V= @(G) G/Chi_eff0+V0;
Nf = length(f); Vth = zeros(1,Nf);
for n=1:Nf
   A=Peaks(:,n)>=-45;
   if max(A)==0
      Vth(n) = inf;
   else
   nth = find(A>0,1,'first');
   Vth(n) = V(nth);
   end
end
fth=f(Vth<inf);
Vth=Vth(Vth<inf);
Nfx=find((fth>98.015e3).*(fth<98.465e3));
Vth(Nfx)=G2V(G(fth(Nfx)*2*pi*1e-3));
P=polyfit(fth,Vth,6); 
Vth_fth = @(f) polyval(P,f);

%% The simulation results (Must run main1_CMT_scan.m first to get the data)
load Sim.mat;

%% Plots
figure(1)
clf;
surf(f*1e-3,V,Peaks);shading interp;view(2);hold on;
plot3(w0_scan/2/pi,Vth_fth(w0_scan/2/pi*1e3),ones(size(w0_scan)),'r--');
xlim([96.315 100.315]);xticks([97:1:100]);
ylim([0.1 0.5]*20);

figure(2)
clf;
XX=x1_max>ini;
surf([Omega_scan(1) Omega_scan(end)]/2e3/pi,[0 100],zeros(2,2));hold on;
surf(Omega_scan/2e3/pi,G_scan*1e-6,abs(XX).');
shading interp;view(2);hold on;
plot3(w0_scan/2/pi,Gth,ones(size(Gth)),'r');
xlim([96.315 100.315]); xticks([97:1:100]);
ylim([00 500]);
% ylabel('Pumping Freq f_p (KHz)'); xlabel('Parametric coefficient (kHz^2)');

figure(3)
clf;
chi=@(w)G(w)./(Vth_fth(w/2/pi*1e3)-V0);
w = (96:0.1:110)*2*pi;
plot(w/2/pi,chi(w),'r');hold on;
% plot([96,100],1/0.84e-3*ones(1,2),'r');
xlim([96.315 100.315]); xticks([97:1:100]);
ylim([0 80]);

Gw2V = @(w,V) chi(w)*(V-V0);
save('Gw2V','Gw2V');

for m=[1,2,3]
    figure(m);
    box on; grid off;
    set(gca, 'Layer', 'top');
    ax = gca;
    ax.LineWidth = 1.2;
    set(gcf,'position',[700,100,300,300]);
    ax.XAxis.MinorTick = 'on'; 
    ax.YAxis.MinorTick = 'on'; 
    ax.TickLength = [0.02, 0.01];
    ax.FontSize = 16;
    % exportgraphics(ax,[num2str(m),'.jpg'],'Resolution',600);
end