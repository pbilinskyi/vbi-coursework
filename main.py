import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import time
import matplotlib

# Set up logging
log_directory = "object_detector_logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Create a timestamp for the log file
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_filename = f"{log_directory}/object_detector_{timestamp}.log"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Also output to console
    ]
)

logger = logging.getLogger(__name__)

matplotlib.set_loglevel('WARNING')  # або 'ERROR' для ще більш суворого фільтрування



class ObjectDetector:
    def __init__(self, search_radius=10, N0=100, M=200, noise_std=0.1,
                 prob_change_state=0.1, object_pos=(5, 1.75*np.pi),
                 object_amplitude_range=(1000,1500), num_object_signals=20, amplitude_range=(200,1200)):
        if amplitude_range is None:
            amplitude_range = [200, 1200]
        self.search_radius = search_radius
        self.N0 = N0
        self.M = M
        self.noise_std = noise_std
        self.prob_change_state = prob_change_state

        self.amplitude_range = amplitude_range
        self.object_pos_radii = object_pos[0]
        self.object_pos_angle = object_pos[1]
        self.num_object_signals = num_object_signals
        self.object_amplitude_range = object_amplitude_range
        self.sigma = 500
        self.current_center = None
        self.center_prev = None

        self.z_grids = []
        self.iteration = 0

        # self.a_param = 3.0 # Можеш поки закоментувати або видалити, якщо характерна довжина його замінить
        self.zgrid_kernel_char_length_sq = (self.search_radius / 5.0)**2 # Наприклад, (10/5)^2 = 4
        self.h_kernel_char_length_sq = (self.search_radius / 5.0)**2   # Або інше значення, наприклад (10/10)^2 = 1
        
        logger.info(f"ObjectDetector initialized with parameters:")
        logger.info(f"  Search radius: {search_radius}")
        logger.info(f"  Initial markers (N0): {N0}")
        logger.info(f"  Maximum markers (M): {M}")
        logger.info(f"  Noise standard deviation: {noise_std}")
        logger.info(f"  State change probability: {prob_change_state}")
        logger.info(f"  Object position (r, θ): ({object_pos[0]}, {object_pos[1]})")
        logger.info(f"  Object amplitude range: {object_amplitude_range}")
        logger.info(f"  Number of object signals: {num_object_signals}")
        logger.info(f"  Background amplitude range: {amplitude_range}")

    def initialize_markers(self):
        """Генерація початкових маркерів (стан 2) з рівномірним розподілом по круговій області."""
        logger.info(f"Initializing {self.N0} markers with state 2 and {self.num_object_signals} object signals")
        
        # Generate background signals (state 2)
        logger.debug("Generating background signals with uniform distribution")
        signals_radii = self.search_radius * np.sqrt(np.random.uniform(0, 1, self.N0))
        signals_angles = np.random.uniform(0, 2 * np.pi, self.N0)
        signals_amplitudes = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], self.N0)
        
        logger.debug(f"Background signals - radii range: [{np.min(signals_radii):.2f}, {np.max(signals_radii):.2f}]")
        logger.debug(f"Background signals - angle range: [{np.min(signals_angles):.2f}, {np.max(signals_angles):.2f}]")
        logger.debug(f"Background signals - amplitude range: [{np.min(signals_amplitudes):.2f}, {np.max(signals_amplitudes):.2f}]")

        # Generate object signals with normal distribution around object position
        logger.debug(f"Generating {self.num_object_signals} object signals around position (r={self.object_pos_radii}, θ={self.object_pos_angle})")
        object_signals_radii = np.random.normal(self.object_pos_radii, 0.1, self.num_object_signals)
        object_signals_angles = np.random.normal(self.object_pos_angle, 0.1, self.num_object_signals)
        object_signals_amplitudes = np.random.uniform(self.object_amplitude_range[0], self.object_amplitude_range[1],
                                                      self.num_object_signals)
        
        logger.debug(f"Object signals - radii range: [{np.min(object_signals_radii):.2f}, {np.max(object_signals_radii):.2f}]")
        logger.debug(f"Object signals - angle range: [{np.min(object_signals_angles):.2f}, {np.max(object_signals_angles):.2f}]")
        logger.debug(f"Object signals - amplitude range: [{np.min(object_signals_amplitudes):.2f}, {np.max(object_signals_amplitudes):.2f}]")

        # Combine background and object signals
        radii = np.concatenate((signals_radii, object_signals_radii))
        angles = np.concatenate((signals_angles, object_signals_angles))
        amplitudes = np.concatenate((signals_amplitudes, object_signals_amplitudes))

        # Convert to Cartesian coordinates
        markers = np.column_stack((
            radii * np.cos(angles),
            radii * np.sin(angles),
            amplitudes
        ))

        # All markers initially have state 2
        states = np.full(self.N0 + self.num_object_signals, 2)

        # Object indices are the last num_object_signals indices
        object_indices = np.arange(self.N0, self.N0 + self.num_object_signals)
        
        logger.info(f"Initialization complete: {len(markers)} total markers created")
        logger.debug(f"Markers shape: {markers.shape}, States shape: {states.shape}")
        
        return markers, states, object_indices

    def generate_new_markers(self, Nk_prev):
        """Генерація нових маркерів (стан 1). Для них амплітуда із діапазону шуму."""
        num_new = self.M - Nk_prev
        logger.info(f"Generating {num_new} new markers with state 1 (total markers will be {self.M})")
        
        if num_new <= 0:
            logger.info(f"No new markers needed, current marker count {Nk_prev} >= max marker count {self.M}")
            return np.array([]), np.array([])
        
        # Generate uniformly distributed angles and radii
        angles = np.random.uniform(0, 2 * np.pi, num_new)
        radii = self.search_radius * np.sqrt(np.random.uniform(0, 1, num_new))
        
        # Generate amplitudes from the background amplitude range
        amplitudes = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], num_new)
        
        logger.debug(f"New markers - radii range: [{np.min(radii):.2f}, {np.max(radii):.2f}]")
        logger.debug(f"New markers - angle range: [{np.min(angles):.2f}, {np.max(angles):.2f}]")
        logger.debug(f"New markers - amplitude range: [{np.min(amplitudes):.2f}, {np.max(amplitudes):.2f}]")
        
        # Convert to Cartesian coordinates
        new_markers = np.column_stack((
            radii * np.cos(angles),
            radii * np.sin(angles),
            amplitudes
        ))
        
        # Calculate sigma (mean amplitude) for later use in weight computation
        self.sigma = np.sum(amplitudes) / len(amplitudes)
        logger.debug(f"Updated sigma value: {self.sigma:.2f}")
        
        logger.info(f"Created {num_new} new markers with state 1")
        return new_markers, np.ones(num_new)  # state 1 is represented as 1

    def update_states(self, states):
        """Оновлення станів маркерів: з ймовірністю prob_change_state маркер змінює стан."""
        logger.info(f"Updating states for {len(states)} markers with change probability {self.prob_change_state}")
        
        # Track initial state counts
        state1_count_before = np.sum(states == 1)
        state2_count_before = np.sum(states == 2)
        logger.debug(f"Before update - State 1: {state1_count_before}, State 2: {state2_count_before}")
        
        # Generate random changes
        state_changes = np.random.random(len(states)) < self.prob_change_state
        num_changes = np.sum(state_changes)
        logger.debug(f"Number of markers changing state: {num_changes} ({num_changes/len(states)*100:.2f}%)")
        
        # Update states - if was 1, becomes 2; if was 2, becomes 1
        states[state_changes] = 3 - states[state_changes]
        
        # Track final state counts
        state1_count_after = np.sum(states == 1)
        state2_count_after = np.sum(states == 2)
        logger.debug(f"After update - State 1: {state1_count_after}, State 2: {state2_count_after}")
        logger.debug(f"Net change - State 1: {state1_count_after - state1_count_before}, State 2: {state2_count_after - state2_count_before}")
        
        return states
    
    def draw_z_grid(self, theta_edges, r_edges, z_grid, iteration=None):
        """
        Візуалізує z_grid у вигляді тепловой карти в полярних координатах
        та ідентифікує потенційні симетричні артефакти.
        
        z_grid: масив розмірності (num_r, num_theta)
        r_edges: масив довжин радіальних меж довжини num_r+1
        theta_edges: масив кутових меж довжини num_theta+1
        iteration: номер ітерації для назви файлу
        """
        # Створюємо 2 полярних графіка: звичайний і нормалізований
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': 'polar'})
        
        # Створюємо координатну сітку для pcolormesh
        Theta, R = np.meshgrid(theta_edges, r_edges)
        
        # 1. Перший графік з оригінальними значеннями
        c1 = ax1.pcolormesh(Theta, R, z_grid, shading='auto', cmap='viridis')
        ax1.set_title("Original z_grid")
        fig.colorbar(c1, ax=ax1, label='z value')
        
        # Додаємо відмітку істинного положення об'єкта для порівняння (тільки для відображення)
        # Це симуляція, тому ми можемо це зробити, в реальному випадку цього не буде
        if hasattr(self, 'object_pos_radii') and hasattr(self, 'object_pos_angle'):
            ax1.scatter(self.object_pos_angle, self.object_pos_radii, 
                    color='red', s=100, marker='*', label='True Position')
            ax1.legend()
        
        # 2. Другий графік з нормалізованими і з порогуванням значеннями для виявлення артефактів
        z_norm = (z_grid - np.min(z_grid)) / (np.max(z_grid) - np.min(z_grid) + 1e-10)
        
        # Знаходимо локальні максимуми в z_grid
        from scipy.ndimage import maximum_filter, gaussian_filter
        from scipy.signal import find_peaks
        
        # Згладжування z_grid для зменшення шуму
        z_smooth = gaussian_filter(z_norm, sigma=1)
        
        # Визначення локальних максимумів
        z_max = maximum_filter(z_smooth, size=3)
        maxima = (z_smooth == z_max) & (z_smooth > 0.7)  # Порогування для знаходження значущих максимумів
        
        # Малюємо нормалізовану теплову карту
        c2 = ax2.pcolormesh(Theta, R, z_smooth, shading='auto', cmap='viridis')
        ax2.set_title("Normalized and Smoothed z_grid")
        fig.colorbar(c2, ax=ax2, label='Normalized z value')
        
        # Знаходимо координати локальних максимумів
        maxima_indices = np.where(maxima)
        if len(maxima_indices[0]) > 0:
            maxima_r = r_edges[maxima_indices[0]]
            maxima_theta = theta_edges[maxima_indices[1]]
            ax2.scatter(maxima_theta, maxima_r, color='red', s=50, marker='x', label='Local Maxima')
            ax2.legend()
            
            # Аналізуємо симетрію локальних максимумів
            if len(maxima_theta) > 1:
                logger.info(f"Found {len(maxima_theta)} local maxima in z_grid")
                for i, (r, theta) in enumerate(zip(maxima_r, maxima_theta)):
                    logger.info(f"Maximum {i+1}: r={r:.2f}, θ={theta:.2f} rad ≈ {np.degrees(theta):.2f}°")
                
                # Перевіряємо на симетричні пари (різниця кутів близька до π)
                for i in range(len(maxima_theta)):
                    for j in range(i+1, len(maxima_theta)):
                        angle_diff = abs(maxima_theta[i] - maxima_theta[j])
                        if abs(angle_diff - np.pi) < 0.3:  # Допуск 0.3 радіан (~17°)
                            logger.warning(f"Potential symmetric artifact detected between maxima {i+1} and {j+1}!")
                            logger.warning(f"Angles: {np.degrees(maxima_theta[i]):.2f}° and {np.degrees(maxima_theta[j]):.2f}° (difference: {np.degrees(angle_diff):.2f}°)")
        
        plt.tight_layout()
        if iteration is not None:
            plt.suptitle(f'z_grid Analysis - Iteration {iteration}', fontsize=16)
            plt.subplots_adjust(top=0.9)
            plt.savefig(f'z_grid_iteration_{iteration}.png')
        plt.show()
    

    def visualize_likelihood(self, markers, states, weights, iteration):
        """Візуалізує функцію правдоподібності для маркерів стану 2"""
        if np.sum(states == 2) == 0:
            logger.warning("No markers in state 2 for likelihood visualization")
            return
            
        state2_mask = states == 2
        markers_state2 = markers[state2_mask]
        x = markers_state2[:, 0]
        y = markers_state2[:, 1]
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        
        # Нормалізуємо ваги для візуалізації
        norm_weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-10)
        
        # Створюємо чотири підграфіки
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Декартові координати з кольоровим маркуванням ваг
        scatter = axs[0, 0].scatter(x, y, c=norm_weights, cmap='viridis', 
                                s=50, alpha=0.7, edgecolors='k')
        axs[0, 0].set_aspect('equal')
        axs[0, 0].set_xlim(-self.search_radius, self.search_radius)
        axs[0, 0].set_ylim(-self.search_radius, self.search_radius)
        axs[0, 0].set_xlabel('X')
        axs[0, 0].set_ylabel('Y')
        axs[0, 0].set_title('State 2 Marker Weights (Cartesian)')
        axs[0, 0].grid(True)
        fig.colorbar(scatter, ax=axs[0, 0], label='Normalized Weight')
        
        # 2. Полярні координати
        ax_polar = plt.subplot(2, 2, 2, projection='polar')
        scatter_polar = ax_polar.scatter(theta, r, c=norm_weights, cmap='viridis',
                                        s=50, alpha=0.7, edgecolors='k')
        ax_polar.set_title('State 2 Marker Weights (Polar)')
        ax_polar.set_rmax(self.search_radius)
        fig.colorbar(scatter_polar, ax=ax_polar, label='Normalized Weight')
        
        # 3. 3D поверхня функції правдоподібності
        # Створюємо сітку для інтерполяції
        grid_size = 50
        x_grid = np.linspace(-self.search_radius, self.search_radius, grid_size)
        y_grid = np.linspace(-self.search_radius, self.search_radius, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Інтерполяція ваг на регулярну сітку за допомогою Радіальних Базисних Функцій
        from scipy.interpolate import Rbf
        if len(x) > 3:  # Потрібно мінімум 3 точки для інтерполяції
            rbf = Rbf(x, y, weights, function='multiquadric', epsilon=2)
            Z = rbf(X, Y)
            
            # Нормалізація для візуалізації
            Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z) + 1e-10)
            
            # 3D Plot
            ax_3d = fig.add_subplot(2, 2, 3, projection='3d')
            surf = ax_3d.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Normalized Likelihood')
            ax_3d.set_title('Likelihood Surface (3D)')
            fig.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=5, label='Normalized Likelihood')
            
            # 4. Контурна карта правдоподібності
            contour = axs[1, 1].contourf(X, Y, Z, 50, cmap='viridis')
            axs[1, 1].set_aspect('equal')
            axs[1, 1].set_xlim(-self.search_radius, self.search_radius)
            axs[1, 1].set_ylim(-self.search_radius, self.search_radius)
            axs[1, 1].set_xlabel('X')
            axs[1, 1].set_ylabel('Y')
            axs[1, 1].set_title('Likelihood Contours')
            axs[1, 1].grid(True)
            fig.colorbar(contour, ax=axs[1, 1], label='Normalized Likelihood')
        else:
            axs[1, 0].text(0.5, 0.5, "Not enough points for interpolation", ha='center', va='center')
            axs[1, 1].text(0.5, 0.5, "Not enough points for interpolation", ha='center', va='center')
        
        plt.tight_layout()
        plt.suptitle(f'Likelihood Visualization - Iteration {iteration}', fontsize=16)
        plt.subplots_adjust(top=0.92)
        plt.savefig(f'likelihood_iteration_{iteration}.png')
        plt.show()


    def compare_position_estimates(self, t):
        """Порівняння максимуму z_grid і центру мас із справжньою позицією об'єкта"""
        # Перевіряємо, чи є поточний центр мас
        if not hasattr(self, 'current_center') or self.current_center is None:
            print(f"Iteration {t}: No current center available")
            return
            
        # Знаходимо максимум у останній z_grid
        if len(self.z_grids) > 0:
            z_grid = self.z_grids[-1]
            max_idx = np.unravel_index(np.argmax(z_grid), z_grid.shape)
            
            # Обчислюємо координати максимуму
            r_edges = np.linspace(0, self.search_radius, 51)
            theta_edges = np.linspace(0, 2 * np.pi, 121)
            r_centers = (r_edges[:-1] + r_edges[1:]) / 2
            theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
            
            zgrid_r = r_centers[max_idx[0]]
            zgrid_theta = theta_centers[max_idx[1]]
            zgrid_x = zgrid_r * np.cos(zgrid_theta)
            zgrid_y = zgrid_r * np.sin(zgrid_theta)
            
            # Центр мас у декартових координатах
            com_x, com_y = self.current_center
            com_r = np.sqrt(com_x**2 + com_y**2)
            com_theta = np.arctan2(com_y, com_x)
            if com_theta < 0:
                com_theta += 2 * np.pi
                
            # Справжня позиція об'єкта
            true_r = self.object_pos_radii 
            true_theta = self.object_pos_angle
            true_x = true_r * np.cos(true_theta)
            true_y = true_r * np.sin(true_theta)
            
            # Створюємо директорію для збереження даних, якщо вона не існує
            output_dir = "position_comparison"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Шлях до файлу
            csv_path = f"{output_dir}/position_comparison.csv"
            file_exists = os.path.exists(csv_path)
            
            # Записуємо результати у CSV
            with open(csv_path, 'a') as f:
                # Заголовок файлу, якщо він новий
                if not file_exists:
                    f.write("iteration,true_r,true_theta,true_x,true_y,")
                    f.write("zgrid_r,zgrid_theta,zgrid_x,zgrid_y,zgrid_max_value,")
                    f.write("com_r,com_theta,com_x,com_y\n")
                    
                # Записуємо дані поточної ітерації
                f.write(f"{t},{true_r:.3f},{true_theta:.3f},{true_x:.3f},{true_y:.3f},")
                f.write(f"{zgrid_r:.3f},{zgrid_theta:.3f},{zgrid_x:.3f},{zgrid_y:.3f},{np.max(z_grid):.3f},")
                f.write(f"{com_r:.3f},{com_theta:.3f},{com_x:.3f},{com_y:.3f}\n")
            
            # Також виведемо основну інформацію в консоль для швидкого аналізу
            print(f"\nПОЗИЦІЇ - Ітерація {t}:")
            print(f"Справжня позиція: [r={true_r:.2f}, θ={np.degrees(true_theta):.1f}°], [x={true_x:.2f}, y={true_y:.2f}]")
            print(f"Максимум z_grid:  [r={zgrid_r:.2f}, θ={np.degrees(zgrid_theta):.1f}°], [x={zgrid_x:.2f}, y={zgrid_y:.2f}]")
            print(f"Центр мас:       [r={com_r:.2f}, θ={np.degrees(com_theta):.1f}°], [x={com_x:.2f}, y={com_y:.2f}]")
            
            # Обчислюємо відстані
            zgrid_error = np.sqrt((zgrid_x - true_x)**2 + (zgrid_y - true_y)**2)
            com_error = np.sqrt((com_x - true_x)**2 + (com_y - true_y)**2)
            zgrid_com_diff = np.sqrt((zgrid_x - com_x)**2 + (zgrid_y - com_y)**2)
            
            print(f"Помилка z_grid: {zgrid_error:.2f}")
            print(f"Помилка центру мас: {com_error:.2f}")
            print(f"Різниця між z_grid і центром мас: {zgrid_com_diff:.2f}")


    def save_weights_to_file(self, current_weights, current_markers_st2, iteration):
        """
        Зберігає ваги та позиції маркерів стану 2 у текстовий файл.
        current_weights: масив нормалізованих ваг для маркерів стану 2.
        current_markers_st2: масив маркерів стану 2 (координати x, y, амплітуда).
        iteration: номер поточної ітерації.
        """
        data_dir = "weights_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created directory {data_dir} for weights data")
    
        file_path = os.path.join(data_dir, f"weights_iteration_{iteration}.txt")
    
        num_st2_markers = len(current_weights)

        if num_st2_markers == 0:
            logger.info(f"Iteration {iteration}: No state 2 markers to save weights for.")
            with open(file_path, 'w') as f:
                f.write(f"# Weights Data for Iteration {iteration}\n")
                f.write("# No State 2 markers found.\n")
            return
        
        # Перевірка узгодженості розмірів
        if num_st2_markers != current_markers_st2.shape[0]:
            logger.error(f"Iteration {iteration}: Mismatch in lengths of weights ({num_st2_markers}) and state 2 markers ({current_markers_st2.shape[0]})!")
            # Можливо, тут варто не зберігати файл або зберегти з помилкою
            with open(file_path, 'w') as f:
                f.write(f"# Weights Data for Iteration {iteration}\n")
                f.write(f"# ERROR: Mismatch in lengths of weights and state 2 markers!\n")
            return

        try:
            with open(file_path, 'w') as f:
                f.write(f"# Weights Data for Iteration {iteration}\n")
                f.write(f"# Number of State 2 markers: {num_st2_markers}\n")
                f.write(f"# Min weight: {np.min(current_weights):.6e}\n")
                f.write(f"# Max weight: {np.max(current_weights):.6e}\n")
                f.write(f"# Sum of weights: {np.sum(current_weights):.6f}\n")
                
                # Ентропія цих нормалізованих ваг (це буде "Entropy - Before" resampling)
                entropy = -np.sum(current_weights * np.log(current_weights + 1e-20)) # Використовуй малий епсилон
                f.write(f"# Entropy of weights: {entropy:.6f}\n\n")
                
                f.write("# Marker_Index_in_St2_Set, X, Y, Amplitude, Weight\n")
                for i in range(num_st2_markers):
                    marker_pos = current_markers_st2[i, :2]
                    marker_amp = current_markers_st2[i, 2]
                    f.write(f"{i}, {marker_pos[0]:.4f}, {marker_pos[1]:.4f}, {marker_amp:.2f}, {current_weights[i]:.6e}\n")
                
                # Додатково, виведемо топ N ваг та позиції їхніх маркерів
                top_n = min(20, num_st2_markers) # Наприклад, топ-20
                # Отримуємо індекси відсортованих ваг (від найбільшої до найменшої)
                sorted_indices = np.argsort(current_weights)[::-1][:top_n] 
                
                f.write(f"\n# Top {top_n} weights and their marker details:\n")
                f.write("# Rank, Original_Marker_Index_in_St2_Set, X, Y, Amplitude, Weight\n")
                for rank, original_idx in enumerate(sorted_indices):
                    marker_pos = current_markers_st2[original_idx, :2]
                    marker_amp = current_markers_st2[original_idx, 2]
                    f.write(f"{rank+1}, {original_idx}, {marker_pos[0]:.4f}, {marker_pos[1]:.4f}, {marker_amp:.2f}, {current_weights[original_idx]:.6e}\n")

                logger.info(f"Successfully saved weights data to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save weights data for iteration {iteration}: {str(e)}")


    def save_z_grid_to_file(self, z_grid, r_edges, theta_edges, iteration):
        """
        Зберігає матрицю z_grid та відповідні координатні сітки у текстовий файл
        для подальшого аналізу.
        
        Parameters:
            z_grid (numpy.ndarray): Матриця інтенсивності
            r_edges (numpy.ndarray): Радіальні межі
            theta_edges (numpy.ndarray): Кутові межі
            iteration (int): Номер ітерації
        """
        # Створюємо директорію для збереження даних, якщо вона не існує
        data_dir = "z_grid_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created directory {data_dir} for z_grid data")
        
        # Формуємо шлях до файлу
        file_path = os.path.join(data_dir, f"z_grid_iteration_{iteration}.txt")
        
        try:
            with open(file_path, 'w') as f:
                # Зберігаємо метадані
                f.write(f"# Z-Grid Data for Iteration {iteration}\n")
                f.write(f"# Shape: {z_grid.shape}\n")
                f.write(f"# Min value: {np.min(z_grid)}\n")
                f.write(f"# Max value: {np.max(z_grid)}\n")
                f.write(f"# Average value: {np.mean(z_grid)}\n\n")
                
                # Зберігаємо координатні сітки
                f.write("# Radial edges (r_edges):\n")
                np.savetxt(f, r_edges, fmt="%.3f")
                f.write("\n# Angular edges (theta_edges):\n")
                np.savetxt(f, theta_edges, fmt="%.3f")
                
                # Зберігаємо матрицю z_grid
                f.write("\n# Z-Grid matrix (rows = radii, columns = angles):\n")
                np.savetxt(f, z_grid, fmt="%.3f")
                
                # Додатково зберігаємо координати з максимальними значеннями
                # Знаходимо топ-10 максимальних значень
                flat_indices = np.argsort(z_grid.flatten())[-10:]  # Індекси топ-10 значень у плоскому масиві
                r_indices, theta_indices = np.unravel_index(flat_indices, z_grid.shape)  # Конвертуємо в 2D індекси
                
                f.write("\n# Top 10 maximum values:\n")
                f.write("# Index_r, Index_theta, r_value, theta_value, z_value\n")
                
                for i, (r_idx, theta_idx) in enumerate(zip(r_indices, theta_indices)):
                    r_val = (r_edges[r_idx] + r_edges[r_idx+1]) / 2  # Середнє значення радіуса для даної комірки
                    theta_val = (theta_edges[theta_idx] + theta_edges[theta_idx+1]) / 2  # Середнє значення кута
                    z_val = z_grid[r_idx, theta_idx]
                    f.write(f"{r_idx}, {theta_idx}, {r_val:.6f}, {theta_val:.6f}, {z_val:.6f}\n")
                
                # Додаємо аналіз симетричних артефактів
                f.write("\n# Analysis of potential symmetric artifacts:\n")
                
                # Знаходимо локальні максимуми
                from scipy.ndimage import maximum_filter, gaussian_filter
                
                # Згладжування та нормалізація для знаходження максимумів
                z_norm = (z_grid - np.min(z_grid)) / (np.max(z_grid) - np.min(z_grid) + 1e-10)
                z_smooth = gaussian_filter(z_norm, sigma=1)
                z_max = maximum_filter(z_smooth, size=3)
                maxima = (z_smooth == z_max) & (z_smooth > 0.7)
                
                maxima_indices = np.where(maxima)
                if len(maxima_indices[0]) > 0:
                    f.write(f"# Found {len(maxima_indices[0])} local maxima\n")
                    
                    # Координати максимумів
                    maxima_r_indices = maxima_indices[0]
                    maxima_theta_indices = maxima_indices[1]
                    
                    # Отримання фактичних значень r і theta
                    maxima_r = np.array([(r_edges[i] + r_edges[i+1])/2 for i in maxima_r_indices])
                    maxima_theta = np.array([(theta_edges[i] + theta_edges[i+1])/2 for i in maxima_theta_indices])
                    
                    f.write("# Local maxima (r, theta, z_value, angle_degrees):\n")
                    for i in range(len(maxima_r)):
                        r_idx, theta_idx = maxima_r_indices[i], maxima_theta_indices[i]
                        r_val = maxima_r[i]
                        theta_val = maxima_theta[i]
                        z_val = z_grid[r_idx, theta_idx]
                        angle_degrees = np.degrees(theta_val)
                        f.write(f"{r_val:.4f}, {theta_val:.4f}, {z_val:.6f}, {angle_degrees:.2f}°\n")
                    
                    # Перевірка на симетричні пари
                    f.write("\n# Potential symmetric pairs (difference close to π):\n")
                    symmetric_pairs_found = False
                    for i in range(len(maxima_theta)):
                        for j in range(i+1, len(maxima_theta)):
                            angle_diff = abs(maxima_theta[i] - maxima_theta[j])
                            angle_diff_norm = min(angle_diff, 2*np.pi - angle_diff)  # Нормалізація різниці кутів
                            
                            if abs(angle_diff_norm - np.pi) < 0.3:  # Допуск 0.3 радіан (~17°)
                                symmetric_pairs_found = True
                                f.write(f"# Pair {i+1} and {j+1}: Angles {np.degrees(maxima_theta[i]):.2f}° and {np.degrees(maxima_theta[j]):.2f}°\n")
                                f.write(f"# - Angular difference: {np.degrees(angle_diff_norm):.2f}° (π = 180°)\n")
                                f.write(f"# - Radii: {maxima_r[i]:.2f} and {maxima_r[j]:.2f}\n")
                                f.write(f"# - Z values: {z_grid[maxima_r_indices[i], maxima_theta_indices[i]]:.6f} " 
                                    f"and {z_grid[maxima_r_indices[j], maxima_theta_indices[j]]:.6f}\n\n")
                    
                    if not symmetric_pairs_found:
                        f.write("# No symmetric pairs found\n")
                else:
                    f.write("# No significant local maxima found\n")
            
            logger.info(f"Successfully saved z_grid data to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save z_grid data: {str(e)}")


    def compute_weights(self, markers, states):
        """Обчислення ваг для маркерів стану 2 з використанням логарифмічної шкали."""
        logger.info("Computing weights for state 2 markers with logarithmic scaling")
        
        # Підготовка даних (той самий код, що й раніше)
        state2_mask = (states == 2)
        if not np.any(state2_mask):
            logger.warning("No markers in state 2, returning empty weights array")
            return np.array([])
        
        markers_state2 = markers[state2_mask]
        x = markers_state2[:, 0]
        y = markers_state2[:, 1]
        amplitudes = markers_state2[:, 2]
        r_signals = np.sqrt(x ** 2 + y ** 2)
        theta_signals = np.arctan2(y, x)
        theta_signals = np.where(theta_signals < 0, theta_signals + 2 * np.pi, theta_signals)
        
        # Параметри сітки та моделі (той самий код)
        num_r = 50
        num_theta = 120
        r_edges = np.linspace(0, self.search_radius, num_r + 1)
        theta_edges = np.linspace(0, 2 * np.pi, num_theta + 1)
        r_centers = (r_edges[:-1] + r_edges[1:]) / 2
        theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
        
        a = 0.005  # параметр для експоненційного ядра
        
        # ===== Обчислення sigma
        # noise_mask = (states != 2)
        # if not np.any(noise_mask):
        #     sigma = 200
        # else:
        #     noise_amplitudes = markers[noise_mask][:, 2]
        #     sigma = np.sum(noise_amplitudes) / len(noise_amplitudes)
        noise_mask = (states != 2)
        if np.any(noise_mask):
            noise_amplitudes = markers[noise_mask][:, 2]
            if len(noise_amplitudes) > 1:
                sigma_A_std = np.std(noise_amplitudes)
                logger.debug(f"Calculated amplitude noise std (sigma_A_std): {sigma_A_std:.2f} from {len(noise_amplitudes)} noise markers.")
            elif len(noise_amplitudes) == 1:
                sigma_A_std = np.abs(noise_amplitudes[0] - np.mean(self.amplitude_range)) / 2.0 # Дуже груба оцінка
                logger.warning(f"Only one noise marker found. sigma_A_std estimated heuristically: {sigma_A_std:.2f}")
            else:
                sigma_A_std = self.noise_std
                logger.warning(f"Unexpected: no noise markers despite noise_mask. Using self.noise_std for sigma_A_std: {sigma_A_std:.2f}")
        else:
            logger.warning(f"No noise markers (state != 2) to estimate amplitude noise std. Using a default value (e.g., self.noise_std or a fraction of object amplitude).")
            sigma_A_std = (self.amplitude_range[1] - self.amplitude_range[0]) / 3.0
            if sigma_A_std == 0: sigma_A_std = self.noise_std # Якщо діапазон нульовий

        sigma_A_std = max(sigma_A_std, 1e-6)

        # =====
        
        I_k = (self.object_amplitude_range[0] + self.object_amplitude_range[1]) / 2
        z_grid = np.zeros((num_r, num_theta))
        R_okil = 4
        
        # Обчислення z_grid (той самий код, як раніше)
        for i in range(num_r):
            for j in range(num_theta):
                r_c = r_centers[i]
                theta_c = theta_centers[j]
                
                d = np.sqrt(
                    r_c ** 2
                    + r_signals ** 2
                    - 2 * r_c * r_signals * np.cos(theta_c - theta_signals)
                )
                
                mask = d <= R_okil
                n_ij = np.count_nonzero(mask)
                if n_ij == 0:
                    continue
                
                A_mask = amplitudes[mask]
                A_avg = A_mask.mean()
                
                # m_k = np.exp(-a * d[mask] ** 2 / (num_r * num_theta))

                # Для m_k:
                m_k = np.exp(-d[mask]**2 / self.zgrid_kernel_char_length_sq)
                
                z_grid[i, j] = (1 / np.sqrt(n_ij)) * np.sum((A_mask - A_avg) * m_k)

        # Крок 2: Згладити z_grid ОДИН РАЗ, ПІСЛЯ того, як вона вся розрахована
        from scipy.ndimage import gaussian_filter # Цей імпорт краще винести на початок файлу
        z_grid_smoothed = gaussian_filter(z_grid, sigma=1.0) # Або спробуй sigma=0.5 для меншого розмиття
        z_grid = z_grid_smoothed
        logger.debug("Applied Gaussian filter to z_grid (once after full calculation).")


        # Додаємо візуалізацію z_grid на кожній 5-й ітерації
        self.draw_z_grid(theta_edges, r_edges, z_grid)
        self.save_z_grid_to_file(z_grid, r_edges, theta_edges, self.iteration)
        
        # Також зберігаємо z_grid для подальшого аналізу            
        self.z_grids.append(z_grid.copy())
        


        # ------- КЛЮЧОВІ ЗМІНИ: ЛОГАРИФМІЧНЕ ОБЧИСЛЕННЯ ВАГ -------
        logger.debug("Computing weights using logarithmic scale")
        log_weights = np.zeros(len(markers_state2))
        
        for idx in range(len(markers_state2)):
            r = r_signals[idx]
            theta = theta_signals[idx]
            
            # Ініціалізуємо логарифм правдоподібності замість самої правдоподібності
            log_L = 0.0
            
            for i in range(num_r):
                for j in range(num_theta):
                    if z_grid[i, j] == 0:
                        continue
                        
                    r_center = r_centers[i]
                    theta_center = theta_centers[j]
                    
                    dist = np.sqrt(r**2 + r_center**2 - 2*r*r_center*np.cos(theta - theta_center))
                    # h_ij = np.exp(-a * dist ** 2 / (num_r * num_theta))

                    
                    # Для h_ij (де dist - це відстань, а не її квадрат):
                    h_ij = np.exp(-dist**2 / self.h_kernel_char_length_sq)
                    
                    if h_ij < 1e-10:
                        continue
                        
                    mu_ij = I_k * h_ij
                    
                    # Замість множення експонент додаємо їх показники
                    # exponent = - (mu_ij * (mu_ij - 2 * z_grid[i, j])) / (2 * sigma ** 2)
                    # exponent = (mu_ij * (mu_ij + 2 * z_grid[i, j])) / (2 * sigma_A_std ** 2) # Попередня версія
                    # exponent = (mu_ij * z_grid[i, j]) / (sigma_A_std ** 2) # НОВА ПРОПОЗИЦІЯ
                    K_scale = 0.01
                    exponent = K_scale * (mu_ij * z_grid[i, j]) / (sigma_A_std ** 2)
                    log_L += exponent  # Ключова зміна тут!
            
            log_weights[idx] = log_L
        
        # ------- НОРМАЛІЗАЦІЯ ЛОГАРИФМІЧНИХ ВАГ -------
        # Використання "log-sum-exp trick" для уникнення переповнення при перетворенні назад
        
        # 1. Знаходимо максимальне значення log_weight
        if len(log_weights) > 0:
            max_log_weight = np.max(log_weights)
            
            # 2. Віднімаємо максимальне значення для числової стабільності
            shifted_log_weights = log_weights - max_log_weight
            
            # 3. Перетворюємо назад до звичайної шкали
            weights = np.exp(shifted_log_weights)
            
            # 4. Нормалізуємо
            total_weight = np.sum(weights)
            if total_weight > 0:
                weights /= total_weight
            else:
                weights = np.ones(len(weights)) / len(weights)
        else:
            weights = np.array([])
        
        logger.debug(f"Log-weights range: [{np.min(log_weights) if len(log_weights) > 0 else 'N/A'}, "
                    f"{np.max(log_weights) if len(log_weights) > 0 else 'N/A'}]")
        logger.debug(f"Normalized weights sum: {np.sum(weights):.6f}")

        # ----- ДОДАЙ ВИКЛИК ЗБЕРЕЖЕННЯ ВАГ ТУТ -----
        if len(weights) > 0 and len(weights) == markers_state2.shape[0]: # Переконайся, що є що зберігати і розміри збігаються
            # states_for_saving - це просто масив двійок такої ж довжини, як weights, він тут не дуже потрібен,
            # але якщо функція збереження його очікує, можна передати.
            # Або модифікуй save_weights_to_file, щоб він приймав лише weights та markers_state2.
            # Я змінив save_weights_to_file вище, щоб він приймав current_markers_st2.
            self.save_weights_to_file(weights, markers_state2, self.iteration)
        elif len(weights) > 0 and len(weights) != markers_state2.shape[0]:
             logger.error(f"Iteration {self.iteration}: Mismatch for saving weights! Weights len: {len(weights)}, Markers_st2 len: {markers_state2.shape[0]}")
        # -------------------------------------------
        
        
        return weights



    def compute_position_gaussian(self, markers, states, weights, t, sigma=0.5):
        """Обчислення позиції центру мас."""
        logger.info(f"Computing position (center of mass) at iteration {t}")
        
        state2_mask = states == 2
        state2_count = np.sum(state2_mask)
        logger.debug(f"Number of markers in state 2: {state2_count}")
        
        if state2_count == 0:
            logger.warning("No markers in state 2, cannot compute center of mass")
            return np.array([0, 0]), 0
        
        markers_state2 = markers[state2_mask]
        # Only x,y coordinates (without amplitude)
        positions = markers_state2[:, :2]
        
        # Compute weighted center of mass
        x_k = np.sum(weights[:, np.newaxis] * positions, axis=0)
        logger.debug(f"Computed center of mass (cartesian): [{x_k[0]:.4f}, {x_k[1]:.4f}]")
        
        # Convert to polar for logging
        r = np.sqrt(x_k[0] ** 2 + x_k[1] ** 2)
        theta = np.arctan2(x_k[1], x_k[0])
        if theta < 0:
            theta += 2 * np.pi
        
        logger.debug(f"Computed center of mass (polar): [r={r:.4f}, θ={theta:.4f} rad ≈ {np.degrees(theta):.2f}°]")
        
        # Calculate distance metric P
        distances = np.sum((positions - x_k) ** 2, axis=1)
        delta_approx = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-distances / (2 * sigma ** 2))
        P = np.sum(delta_approx)
        
        logger.debug(f"Confidence metric P: {P}")
        
        return x_k, P


    def compute_position_exact(self, markers, states, weights, t):
        """
        Обчислення положення об'єкту за алгоритмом статті.
        """

        state2_mask = states == 2
        markers_state2 = markers[state2_mask]
        positions = markers_state2[:, :2]

        # Обчислюємо положення об'єкту (центр мас)
        x_k = np.sum(weights[:, np.newaxis] * positions, axis=0)

        P = 0

        for pos in positions:
            # D(x,y) = 1 якщо x=y, інакше 0
            if np.allclose(pos, x_k, atol=1e-10):
                P += 1
        x, y = x_k
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        if theta < 0:
            theta += 2 * np.pi
        x_k_polar = (r,theta)

        if(t%5 ==0):
            pass
        if(P==1):
            pass
        return x_k, P


    def multinomial_resampling(self, markers, weights):
        """
        Мультиноміальний ресемплінг:
          - Для кожного з N нових маркерів генерується випадкове число r.
          - Вибирається індекс j, такий, що r потрапляє у відповідний інтервал
            [cumsum[j-1], cumsum[j]).
          - Новий маркер копіюється з markers[j].
        """
        N = len(markers)
        # Накопичена сума ваг
        cumulative_sum = np.cumsum(weights)
        # Виправляємо можливі похибки, щоб останній елемент точно дорівнював 1
        cumulative_sum[-1] = 1.0

        new_markers = np.zeros_like(markers)
        for i in range(N):
            r = np.random.rand()  # випадкове число від 0 до 1
            idx = np.searchsorted(cumulative_sum, r)
            new_markers[i] = markers[idx]
        return new_markers


    def resample_markers(self, markers, weights, states):
        indices_state_2 = np.where(states == 2)[0]
        markers_state_2 = markers[indices_state_2]

        if len(markers_state_2) == 0 or len(weights) == 0:
            return np.array([])

        if len(markers_state_2) != len(weights):
            raise ValueError("Кількість маркерів та ваг повинна бути однаковою")

        if not np.isclose(np.sum(weights), 1.0, rtol=1e-5):
            total = np.sum(weights)
            if total > 0:
                weights = weights / total
            else:
                weights = np.ones_like(weights) / len(weights)

        N_k = len(markers_state_2)
        indices = np.random.choice(N_k, size=N_k, replace=True, p=weights)
        new_markers_state_2 = markers_state_2[indices].copy()

        new_markers = markers.copy()
        new_markers[indices_state_2] = new_markers_state_2
        new_weights = np.ones(N_k) / N_k

        return new_markers, new_weights


    def resample_step(self, markers, weights, states):
        """Ресемплінг маркерів стану 2."""
        logger.info(f"Performing resampling of markers")
        
        indices_state_2 = np.where(states == 2)[0]
        state2_count = len(indices_state_2)
        logger.debug(f"Number of markers in state 2: {state2_count}")
        
        if state2_count == 0:
            logger.warning("No markers in state 2, skipping resampling")
            return markers, np.array([])
        
        markers_state_2 = markers[indices_state_2]
        
        # Check weights
        if len(weights) == 0:
            logger.warning("Empty weights array, skipping resampling")
            return markers, np.array([])
        
        if len(markers_state_2) != len(weights):
            error_msg = f"Number of markers ({len(markers_state_2)}) and weights ({len(weights)}) must be the same"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Normalize weights if needed
        if not np.isclose(np.sum(weights), 1.0, rtol=1e-5):
            total = np.sum(weights)
            logger.debug(f"Normalizing weights. Sum before normalization: {total:.6f}")
            
            if total > 0:
                weights = weights / total
            else:
                logger.warning("Sum of weights is zero or negative, setting uniform weights")
                weights = np.ones_like(weights) / len(weights)
        
        # Perform multinomial resampling
        logger.debug("Performing multinomial resampling")
        new_markers_state_2 = self.multinomial_resampling(markers_state_2, weights)
        
        # Log weight distribution before and after resampling
        before_entropy = -np.sum(weights * np.log(weights + 1e-10))
        after_weights = np.ones(len(new_markers_state_2)) / len(new_markers_state_2)
        after_entropy = -np.sum(after_weights * np.log(after_weights + 1e-10))
        
        logger.debug(f"Weight entropy - Before: {before_entropy:.4f}, After: {after_entropy:.4f}")
        
        # Replace old markers with resampled ones
        new_markers = markers.copy()
        new_markers[indices_state_2] = new_markers_state_2
        
        logger.debug(f"Resampling complete, returned {len(new_markers)} markers with {len(weights)} weights")
        
        return new_markers, weights

    def clip_radius(self, markers, max_radius):
        r = np.linalg.norm(markers, axis=1)

        mask = r > max_radius
        if np.any(mask):
            markers[mask] *= max_radius / r[mask, np.newaxis]

        return markers


    def update_markers(self, markers, states, dt, current_center, center_prev, t, center_prev2=None):
        logger.info(f"Updating marker positions at iteration {t}")
        
        state2_mask = (states == 2)
        state2_count = np.sum(state2_mask)
        logger.debug(f"Number of markers in state 2: {state2_count}")

        if np.any(state2_mask):
            logger.debug(f"Processing markers in state 2")
            
            # Логування інформації про маркери для діагностики
            x_vals, y_vals = markers[state2_mask, 0], markers[state2_mask, 1]
            r_vals = np.sqrt(x_vals ** 2 + y_vals ** 2)
            theta_vals = np.arctan2(y_vals, x_vals)
            
            logger.debug(f"Marker positions - x range: [{np.min(x_vals):.4f}, {np.max(x_vals):.4f}], " 
                        f"y range: [{np.min(y_vals):.4f}, {np.max(y_vals):.4f}]")
            logger.debug(f"Marker positions (polar) - r range: [{np.min(r_vals):.4f}, {np.max(r_vals):.4f}], " 
                        f"θ range: [{np.min(theta_vals):.4f}, {np.max(theta_vals):.4f}]")
            
            # Логування поточного центру в обох координатних системах
            current_center_r = np.sqrt(current_center[0]**2 + current_center[1]**2)
            current_center_theta = np.arctan2(current_center[1], current_center[0])
            
            logger.debug(f"Current center (cartesian): [{current_center[0]:.4f}, {current_center[1]:.4f}]")
            logger.debug(f"Current center (polar): [r={current_center_r:.4f}, θ={current_center_theta:.4f} rad]")
            
            # Обчислення швидкості у декартових координатах
            if center_prev is not None:
                logger.debug(f"Previous center (cartesian): [{center_prev[0]:.4f}, {center_prev[1]:.4f}]")
                
                # Також розрахуємо і полярні координати попереднього центру (лише для логування)
                center_prev_r = np.sqrt(center_prev[0]**2 + center_prev[1]**2)
                center_prev_theta = np.arctan2(center_prev[1], center_prev[0])
                logger.debug(f"Previous center (polar): [r={center_prev_r:.4f}, θ={center_prev_theta:.4f} rad]")
                
                # ВИПРАВЛЕНО: Тепер використовуємо правильну формулу для швидкості в декартових координатах
                if center_prev2 is not None and t > 1:
                    # Використовуємо 3-точкову формулу (21) з статті Васіна
                    logger.debug("Using 3-point velocity estimation formula")
                    velocity = (3 * current_center - 4 * center_prev + center_prev2) / (2 * dt)
                    logger.debug(f"3-point estimated velocity (cartesian): [{velocity[0]:.4f}, {velocity[1]:.4f}]")
                else:
                    # Використовуємо 2-точкову формулу (22) з статті Васіна
                    logger.debug("Using 2-point velocity estimation with cartesian coordinates")
                    velocity = (current_center - center_prev) / dt
                    logger.debug(f"2-point estimated velocity (cartesian): [{velocity[0]:.4f}, {velocity[1]:.4f}]")
                    
                # Логування - порівнюємо з попередньою некоректною формулою для відстеження змін
                # old_velocity_r = (current_center_r - center_prev[0]) / dt
                # old_velocity_theta = (current_center_theta - center_prev[1]) / dt
                # logger.debug(f"OLD INCORRECT velocity calculation would be: [{old_velocity_r:.4f}, {old_velocity_theta:.4f}]")
                # logger.debug(f"Coordinate system mixing error fixed!")
            else:
                logger.debug("First iteration or no previous center, velocity set to 0")
                velocity = np.zeros(2)

            # Додаємо шум до всіх маркерів стану 2
            noise = np.random.normal(0, self.noise_std, size=(np.sum(state2_mask), 2))
            logger.debug(f"Generated noise with mean: {np.mean(noise):.6f}, std: {np.std(noise):.6f}")
            
            # Застосовуємо шум
            markers[state2_mask, :2] += noise * dt
            logger.debug(f"Added noise to markers")
            
            # ВИПРАВЛЕНО: Тепер застосовуємо швидкість до маркерів стану 2, якщо маємо попередній центр
            if center_prev is not None:
                logger.debug(f"Applying velocity to markers: [{velocity[0]:.4f}, {velocity[1]:.4f}]")
                markers[state2_mask, :2] += velocity * dt
                logger.debug(f"Applied velocity to markers")
            
            # Обмежуємо радіус маркерів
            markers[state2_mask, :2] = self.clip_radius(markers[state2_mask, :2], self.search_radius)
            logger.debug(f"Clipped markers to radius {self.search_radius}")
        else:
            logger.warning("No state 2 markers found, skipping position update")

        logger.debug(f"Marker update completed for iteration {t}")
        return markers


    def move_clockwise(self, markers, dt, angular_speed):
        # markers – np.array з першим і другим стовпцями як координати x та y
        r = np.sqrt(markers[:, 0] ** 2 + markers[:, 1] ** 2)
        theta = np.arctan2(markers[:, 1], markers[:, 0])
        # Обертання за часовою стрілкою – віднімання кута
        theta_new = theta - angular_speed * dt
        markers[:, 0] = r * np.cos(theta_new)
        markers[:, 1] = r * np.sin(theta_new)
        return markers

    def move_noise_markers(self, markers, states, dt, noise_std=0.1):
        """
        Додає невеликий випадковий рух тільки до маркерів стану 1.
        Маркери стану 2 залишаються незмінними.
        """
        updated_markers = markers.copy()

        state1_mask = states == 1

        if np.any(state1_mask):
            markers_state1 = updated_markers[state1_mask]

            r = np.sqrt(markers_state1[:, 0] ** 2 + markers_state1[:, 1] ** 2)
            theta = np.arctan2(markers_state1[:, 1], markers_state1[:, 0])

            r_noise = np.random.normal(0, noise_std, size=len(markers_state1))
            r = np.clip(r + r_noise * dt, 0, self.search_radius)

            theta_noise = np.random.normal(0, noise_std, size=len(markers_state1))
            theta += theta_noise * dt

            markers_state1[:, 0] = r * np.cos(theta)
            markers_state1[:, 1] = r * np.sin(theta)

            updated_markers[state1_mask] = markers_state1

        return updated_markers

    def move_object_signals(self, markers, object_mask, angular_speed, dt):
        obj_markers = markers[object_mask].copy()

        x = obj_markers[:, 0]
        y = obj_markers[:, 1]
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)

        theta_new = theta - angular_speed * dt

        x_new = r * np.cos(theta_new)
        y_new = r * np.sin(theta_new)

        obj_markers[:, 0] = x_new
        obj_markers[:, 1] = y_new

        markers[object_mask] = obj_markers
        return markers


    def run_detection(self, num_iterations=200, dt=1):
        """Головний цикл детекції з правильним балансом маркерів стану 1 і 2"""
        logger.info(f"Starting detection process with {num_iterations} iterations")
        
        markers, states, object_indices = self.initialize_markers()
        logger.info(f"Initialized {len(markers)} markers: {np.sum(states==1)} in state 1, {np.sum(states==2)} in state 2")

        for t in range(num_iterations):
            self.iteration = t
            logger.info(f"\n{'='*20} ITERATION {t} {'='*20}")
            
            # Генерація нових маркерів стану 1
            new_markers, new_states = self.generate_new_markers(len(markers))
            if len(new_markers) > 0:
                markers = np.vstack((markers, new_markers))
                states = np.hstack((states, new_states))
                logger.debug(f"Added {len(new_markers)} new markers with state 1, total markers: {len(markers)}")
                
            if t == 0:
                self.visualize(markers, states, t)
                
            # Оновлення станів маркерів
            states = self.update_states(states)
            
            # Обчислення ваг для маркерів стану 2
            weights = self.compute_weights(markers, states)
            
            # Обчислення позиції центру мас
            self.current_center, P = self.compute_position_gaussian(markers, states, weights, t)

            # ДОДАНО: Порівняння оцінок позиції
            self.compare_position_estimates(t)
            
            # Ресемплінг маркерів стану 2
            markers, weights = self.resample_step(markers, weights, states)
            
            # Оновлення позицій маркерів
            markers = self.update_markers(markers, states, dt, self.current_center, self.center_prev, t)
            self.center_prev = self.current_center
            
            if t % 1 == 0 or P >= 0.9:
                self.visualize(markers, states, t, self.current_center)
                
                # Візуалізуємо функцію правдоподібності
                self.visualize_likelihood(markers, states, weights, t)
            
            # FIXED: Замість видалення всіх маркерів стану 1, обмежуємо загальну кількість
            # з балансом між маркерами стану 1 і 2
            state1_mask = states == 1
            state2_mask = states == 2
            state1_count = np.sum(state1_mask)
            state2_count = np.sum(state2_mask)
            
            logger.debug(f"Before marker management: {state1_count} in state 1, {state2_count} in state 2, total: {len(markers)}")
            
            # Якщо загальна кількість маркерів перевищує ліміт, потрібно зменшити їх кількість
            if state1_count + state2_count > self.M:
                # Обчислюємо цільовий розподіл маркерів: 30% стану 1, 70% стану 2
                target_state1_count = int(self.M * 0.3)
                target_state2_count = self.M - target_state1_count
                
                # Індекси маркерів стану 1 і 2
                state1_indices = np.where(state1_mask)[0]
                state2_indices = np.where(state2_mask)[0]
                
                # Випадково відбираємо маркери для збереження
                if state1_count > target_state1_count:
                    keep_state1_indices = np.random.choice(state1_indices, target_state1_count, replace=False)
                else:
                    keep_state1_indices = state1_indices
                
                if state2_count > target_state2_count:
                    keep_state2_indices = np.random.choice(state2_indices, target_state2_count, replace=False)
                else:
                    keep_state2_indices = state2_indices
                
                # Об'єднуємо індекси маркерів для збереження
                keep_indices = np.concatenate((keep_state1_indices, keep_state2_indices))
                
                # Оновлюємо маркери і стани
                markers = markers[keep_indices]
                states = states[keep_indices]
                
                logger.debug(f"Limited markers to {len(markers)}: {np.sum(states==1)} in state 1, {np.sum(states==2)} in state 2")
            else:
                logger.debug(f"No need to limit markers, current count: {len(markers)}")

        logger.info(f"Detection completed after {num_iterations} iterations")
        return markers, states



    def visualize(self, markers, states, iteration, center_of_mass=None):
        """Візуалізація результатів у декартових та полярних координатах"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        x = markers[:, 0]
        y = markers[:, 1]
        weights = markers[:, 2]
        norm_weights = 10 + (weights - 400) / (1600 - 400) * (100 - 10)

        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        theta_deg = np.degrees(np.where(theta < 0, theta + 2 * np.pi, theta))

        # Діаграма: кут (у градусах) проти радіусу
        for state, color, alpha, label in [(1, 'gray', 0.5, 'State 1'),
                                           (2, 'blue', 1.0, 'State 2')]:
            mask = states == state
            ax1.scatter(theta_deg[mask], r[mask], color=color,
                        alpha=alpha, label=label, s=5)

        # Додаємо центр мас, якщо він переданий
        if center_of_mass is not None:
            # Перетворюємо координати центру мас у полярні
            com_x, com_y = center_of_mass
            com_r = np.sqrt(com_x ** 2 + com_y ** 2)
            com_theta = np.arctan2(com_y, com_x)
            com_theta_deg = np.degrees(com_theta if com_theta >= 0 else com_theta + 2 * np.pi)

            # Додаємо червону точку для центру мас
            ax1.scatter(com_theta_deg, com_r, color='red', s=100, marker='*',
                        label='Центр мас', edgecolor='black', zorder=10)

        ax1.set_xlabel('Кут (градуси)')
        ax1.set_ylabel('Радіус')
        ax1.set_title(f'Ітерація {iteration}')
        ax1.set_xlim(0, 360)
        ax1.set_ylim(0, self.search_radius)
        ax1.grid(True)
        ax1.legend()
        ax1.set_xticks(np.arange(0, 360, 30))

        ax2 = plt.subplot(122, projection='polar')
        for state, color, alpha, label in [(1, 'gray', 0.5, 'Стан 1'),
                                           (2, 'blue', 1.0, 'Стан 2')]:
            mask = states == state
            ax2.scatter(theta[mask], r[mask], color=color,
                        alpha=alpha, label=label, s=norm_weights[mask]/10)

        if center_of_mass is not None:
            ax2.scatter(com_theta, com_r, color='red', s=100, marker='*',
                        label='Центр мас', edgecolor='black', zorder=10)

        ax2.set_rmax(self.search_radius)
        ax2.set_title('Полярні координати')
        ax2.grid(True)
        ax2.set_thetagrids(np.arange(0, 360, 30))
        ax2.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    detector = ObjectDetector(
        search_radius=10,
        N0=500,
        M=1000,
        noise_std=0.1)
    markers, states = detector.run_detection()
