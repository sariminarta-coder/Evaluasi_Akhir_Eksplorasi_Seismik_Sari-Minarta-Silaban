clc;
clear all;
close all;

% 1. Definisi Fungsi (Persamaan No. 3)
% f(x1, x2) = 0.25x1 + 5x1^2 + x1^4 - 9x1^2x2 + 3x2^2 + 2x2^4
f = @(x) 0.25*x(1) + 5*x(1)^2 + x(1)^4 - 9*x(1)^2*x(2) + 3*x(2)^2 + 2*x(2)^4;
N = 2; % Dimensi masalah

% 2. Inisialisasi Simplex Awal (3 Titik Manual)
% Kita tentukan koordinat segitiga awal secara eksplisit
P = zeros(N, N+1); 
P(:, 1) = [-1.8; 2.2]; % Titik 1 (Start Utama)
P(:, 2) = [-1.4; 2.2]; % Titik 2 (Geser X)
P(:, 3) = [-1.8; 2.6]; % Titik 3 (Geser Y)

F = [f(P(:, 1)), f(P(:, 2)), f(P(:, 3))]; % Nilai fungsi awal

% Parameter Algoritma Nelder-Mead
alpha = 1; gamma = 2; rho = 0.5; sigma = 0.5;

% Kriteria Berhenti
max_iter = 1000;
tol_f = 1e-8; 
tol_x = 1e-8; 

% 3. Inisialisasi Rekaman Data
history = zeros(max_iter, 5); % [iter, x1_best, x2_best, fval_best, RMS]
simplex_history = cell(max_iter, 1);
iter = 0;

fprintf('==================================================\n');
fprintf('START OPTIMASI NELDER-MEAD (PERSAMAAN 3)\n');
fprintf('Titik Awal Simplex:\n');
disp(P);
fprintf('==================================================\n');

% 4. LOOP OPTIMASI UTAMA (Algoritma Manual)
while iter < max_iter
    iter = iter + 1;

    % a. Urutkan Simplex: Cari B (Best), G (Good), W (Worst)
    [F_sorted, idx] = sort(F);
    PB = P(:, idx(1)); % P Best
    PG = P(:, idx(2)); % P Good
    PW = P(:, idx(3)); % P Worst
    FB = F_sorted(1);  
    FW = F_sorted(3);  
    
    PC = (PB + PG) / N; % Centroid

    % b. Reflection
    PR = PC + alpha * (PC - PW);
    FR = f(PR);

    if FR < FB
        % c. Expansion
        PE = PC + gamma * (PR - PC);
        FE = f(PE);
        if FE < FR
            P(:, idx(3)) = PE; F(idx(3)) = FE;
        else
            P(:, idx(3)) = PR; F(idx(3)) = FR;
        end
    elseif FR < F_sorted(2) % FR < FG
        % d. Accept Reflection
        P(:, idx(3)) = PR;
        F(idx(3)) = FR;
    else 
        % e. Contraction
        if FR < FW 
            PK = PC + rho * (PR - PC); % Outside
        else 
            PK = PC + rho * (PW - PC); % Inside
        end
        FK = f(PK);
        
        if FK < FW
            P(:, idx(3)) = PK; F(idx(3)) = FK;
        else
            % f. Shrink
            P(:, idx(2)) = PB + sigma * (P(:, idx(2)) - PB); 
            P(:, idx(3)) = PB + sigma * (P(:, idx(3)) - PB); 
            F(idx(2)) = f(P(:, idx(2)));
            F(idx(3)) = f(P(:, idx(3)));
        end
    end

    % 5. Penyimpanan Data
    P_avg = mean(P, 2);
    RMS = sqrt(sum(sum((P - P_avg).^2)) / N); % Ukuran simplex
    
    history(iter, 1:4) = [iter, PB(1), PB(2), FB];
    history(iter, 5) = RMS;
    simplex_history{iter} = P;
    
    % 6. Cek Konvergensi
    if (abs(FW - FB) < tol_f) && (RMS < tol_x)
        break;
    end
end

% Rapikan data
history = history(1:iter, :);
x_optimal = P(:, idx(1));
fval = F(idx(1));

% 7. Tampilkan Hasil
fprintf('Hasil Akhir NM:\n');
fprintf('  Iterasi: %d\n', iter);
fprintf('  Min (x1, x2): %.6f, %.6f\n', x_optimal(1), x_optimal(2));
fprintf('  Nilai f(x): %.10f\n', fval);

% =================================================================
% VISUALISASI "TERBAIK" (Style Referensi)
% =================================================================

% Grid Data untuk Plotting
x_plot = linspace(-2.5, 2.5, 100);
y_plot = linspace(-1.5, 3.5, 100);
[X, Y] = meshgrid(x_plot, y_plot);
Z = 0.25*X + 5*X.^2 + X.^4 - 9*X.^2.*Y + 3*Y.^2 + 2*Y.^4;

% --- FIGURE 1: Surface & Contour ---
figure('Name', 'NM: Surface & Contour', 'Position', [100, 100, 1000, 450]);

% Subplot 1: Surface 3D
subplot(1,2,1);
surf(X, Y, Z, 'EdgeColor', 'none');
colormap('parula');
% Trik Visual: Batasi Z-axis supaya lembah terlihat jelas (tidak gepeng)
z_limit = 50; 
zlim([-5, z_limit]); caxis([-5, z_limit]); 
hold on;
plot3(x_optimal(1), x_optimal(2), fval, 'r*', 'MarkerSize', 10, 'LineWidth', 2);
title('Surface Plot (Z-Limited)');
xlabel('x1'); ylabel('x2'); zlabel('f(x)');
view(-30, 45);

% Subplot 2: Contour Logaritmik (Style Referensi)
subplot(1,2,2);
% Gunakan logspace untuk level kontur agar lembah dalam terlihat detail
min_z = min(min(Z));
v_levels = [linspace(min_z, 0, 10), logspace(0, 2, 20)]; % Hybrid levels
contour(X, Y, Z, v_levels, 'LineWidth', 0.8);
hold on;
plot(x_optimal(1), x_optimal(2), 'r*', 'MarkerSize', 12, 'LineWidth', 2);
title('Kontur Plot (Hybrid Levels)');
xlabel('x1'); ylabel('x2');
colorbar; grid on;

% --- FIGURE 2: Jejak Simplex & Konvergensi ---
figure('Name', 'NM: Jejak & Konvergensi', 'Position', [100, 600, 1000, 450]);

% Subplot 1: Jejak Optimasi
subplot(1,2,1);
contour(X, Y, Z, v_levels, 'LineWidth', 0.5); hold on;
% Gambar jalur terbaik
plot(history(:,2), history(:,3), 'k.-', 'LineWidth', 1.5, 'MarkerSize', 8);
% Gambar beberapa segitiga simplex (Awal, Tengah, Akhir)
idx_show = unique(round(linspace(1, iter, 10)));
for i = idx_show
    s = simplex_history{i};
    plot([s(1,:), s(1,1)], [s(2,:), s(2,1)], 'r-', 'LineWidth', 0.5);
end
plot(x_optimal(1), x_optimal(2), 'p', 'MarkerSize', 15, 'MarkerFaceColor','y', 'MarkerEdgeColor','k');
title('Jejak Simplex (Merah) & Pusat (Hitam)');
xlabel('x1'); ylabel('x2');
axis([-2.5 2.5 -1.5 3.5]);

% Subplot 2: Grafik Konvergensi (Semilogy)
subplot(1,2,2);
semilogy(history(:,1), abs(history(:,4)), 'b-', 'LineWidth', 2); hold on;
semilogy(history(:,1), history(:,5), 'r--', 'LineWidth', 1.5);
title('Grafik Konvergensi');
xlabel('Iterasi'); ylabel('Nilai (Log Scale)');
legend('Nilai Fungsi (f)', 'Ukuran Simplex (RMS)');
grid on;