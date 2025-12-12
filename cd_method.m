clc; clear; close all;

% --- 1. Definisi Fungsi & Gradien ---
% Fungsi Objektif (Persamaan 3)
f = @(x) 0.25*x(1) + 5*x(1)^2 + x(1)^4 - 9*x(1)^2*x(2) + 3*x(2)^2 + 2*x(2)^4;

% Gradien (Turunan Parsial Manual)
% g = [df/dx1; df/dx2]
grad_f = @(x) [0.25 + 10*x(1) + 4*x(1)^3 - 18*x(1)*x(2); ...
               -9*x(1)^2 + 6*x(2) + 8*x(2)^3];

% --- 2. Inisialisasi ---
x_curr = [-1.8; 2.2]; % Titik Awal (Samakan dengan NM)
max_iter = 100;
tol = 1e-7;

% Hitung Gradien Awal
g_curr = grad_f(x_curr);
d_curr = -g_curr; % Arah awal (Steepest Descent)

% Simpan History untuk Plotting [iter, x1, x2, f(x)]
history_cg = [];
history_cg = [history_cg; 0, x_curr(1), x_curr(2), f(x_curr)];

fprintf('=== START CONJUGATE GRADIENT (CG) ===\n');
fprintf('Titik Awal: [%.4f, %.4f]\n', x_curr(1), x_curr(2));

% --- 3. Loop Optimasi CG (Fletcher-Reeves) ---
for k = 1:max_iter
    x_prev = x_curr;
    
    % A. Line Search (Mencari langkah optimal 'alpha')
    % Meminimalkan fungsi di sepanjang arah d_curr
    fun_alpha = @(a) f(x_curr + a * d_curr);
    alpha = fminbnd(fun_alpha, 0, 1); % Batas pencarian 0 s.d 1
    
    % B. Update Posisi
    x_next = x_curr + alpha * d_curr;
    
    % Simpan Jejak
    history_cg = [history_cg; k, x_next(1), x_next(2), f(x_next)];
    
    % Cek Konvergensi
    if norm(x_next - x_curr) < tol
        fprintf('Konvergensi tercapai pada iterasi ke-%d\n', k);
        break;
    end
    
    % C. Hitung Gradien Baru
    g_next = grad_f(x_next);
    
    % D. Hitung Beta (Rumus Fletcher-Reeves)
    % Beta mengontrol seberapa besar pengaruh arah sebelumnya
    beta = (g_next' * g_next) / (g_curr' * g_curr);
    
    % E. Update Arah (Conjugate Direction)
    d_next = -g_next + beta * d_curr;
    
    % Persiapan iterasi selanjutnya
    x_curr = x_next;
    g_curr = g_next;
    d_curr = d_next;
end

final_pos = x_curr;
final_val = f(final_pos);

fprintf('-------------------------------------------------\n');
fprintf('Hasil Akhir CG:\n');
fprintf('  Iterasi Total: %d\n', k);
fprintf('  Min (x1, x2) : %.6f, %.6f\n', final_pos(1), final_pos(2));
fprintf('  Nilai f(x)   : %.10f\n', final_val);
fprintf('-------------------------------------------------\n');

% --- 4. VISUALISASI HASIL (Gaya Laporan Lengkap) ---

% Persiapan Grid
x_plot = linspace(-2.5, 2.5, 150);
y_plot = linspace(-1.5, 3.5, 150);
[X, Y] = meshgrid(x_plot, y_plot);
Z = 0.25*X + 5*X.^2 + X.^4 - 9*X.^2.*Y + 3*Y.^2 + 2*Y.^4;

% --- FIGURE 1: Surface Plot (3D) ---
figure('Name', 'CG: 3D Surface', 'Color', 'w', 'Position', [100, 400, 600, 500]);
surf(X, Y, Z, 'EdgeColor', 'none'); 
colormap('parula');
zlim([-5, 50]); caxis([-5, 50]); % Batasi tinggi Z agar lembah terlihat
hold on;
plot3(final_pos(1), final_pos(2), final_val, 'rp', 'MarkerSize', 15, 'MarkerFaceColor','y', 'MarkerEdgeColor','k');
title('Surface Plot: Topologi Fungsi Persamaan 3');
xlabel('x1'); ylabel('x2'); zlabel('f(x)');
view(-25, 50); grid on;


% --- FIGURE 2: Jejak Lintasan & Konvergensi (GABUNGAN) ---
% Ini yang kamu minta: Ada jejak di kiri, grafik di kanan
figure('Name', 'CG: Jejak & Konvergensi', 'Color', 'w', 'Position', [750, 400, 1000, 500]);

% [SUBPLOT KIRI] Peta Kontur + Jejak Lintasan (Trajectory)
subplot(1, 2, 1);
min_z = min(min(Z));
% Level kontur hybrid (Rapat di lembah, renggang di atas)
levels = [linspace(min_z, 0, 15), logspace(0, 2.5, 20)];
contour(X, Y, Z, levels, 'LineWidth', 0.8); hold on;

% Gambar Start
plot(history_cg(1,2), history_cg(1,3), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 8);
text(history_cg(1,2)+0.1, history_cg(1,3), 'Start', 'FontWeight','bold');

% GAMBAR JEJAK LINTASAN (Trajectory) - Ekuivalen "Jejak Simplex"
% Garis biru tebal dengan titik di setiap iterasi
plot(history_cg(:,2), history_cg(:,3), 'b.-', 'LineWidth', 1.5, 'MarkerSize', 10);

% Gambar Finish
plot(final_pos(1), final_pos(2), 'p', 'MarkerSize', 18, 'MarkerFaceColor','y', 'MarkerEdgeColor','k');
text(final_pos(1)+0.1, final_pos(2), 'Min', 'FontWeight','bold');

title({'Jejak Lintasan (Trajectory) CG', 'Smooth Curve (Bukan Zig-Zag)'});
xlabel('x1'); ylabel('x2');
grid on; axis equal;
xlim([-2.5, 2.5]); ylim([-1.5, 3.5]);

% [SUBPLOT KANAN] Grafik Konvergensi
subplot(1, 2, 2);
iterasi = history_cg(:,1);
nilai_f = abs(history_cg(:,4)); % Nilai mutlak f(x)

semilogy(iterasi, nilai_f, 'b-o', 'LineWidth', 2, 'MarkerSize', 4);
grid on;
title('Kecepatan Konvergensi (Log Scale)');
xlabel('Iterasi'); ylabel('|f(x)|');
legend('Nilai Error Fungsi');