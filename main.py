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
    
    def draw_z_grid(self, theta_edges, r_edges, z_grid):
        #   z_grid: масив розмірності (num_r, num_theta)
        #   r_edges: масив довжин радіальних меж довжини num_r+1
        #   theta_edges: масив кутових меж довжини num_theta+1

        # Створюємо полярний графік
        # pcolormesh очікує координати кутів та радіусів у вигляді сітки:
        # Правильно згенеровані координати для pcolormesh
        Theta, R = np.meshgrid(theta_edges, r_edges)

        # Малюємо полярну діаграму
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        c = ax.pcolormesh(Theta, R, z_grid, shading='auto')
        ax.set_title("Polar Heatmap of z_grid")
        fig.colorbar(c, ax=ax, label='z value')

        plt.show()


    def compute_weights(self, markers, states):
        logger.info("Computing weights for state 2 markers")
        
        state2_mask = (states == 2)
        if not np.any(state2_mask):
            logger.warning("No markers in state 2, returning empty weights array")
            return np.array([])
        
        markers_state2 = markers[state2_mask]
        logger.debug(f"Processing {len(markers_state2)} markers in state 2")

        # Перетворюємо координати в полярні
        x = markers_state2[:, 0]
        y = markers_state2[:, 1]
        amplitudes = markers_state2[:, 2]
        logger.debug(f"Amplitudes range: [{np.min(amplitudes):.2f}, {np.max(amplitudes):.2f}], mean: {np.mean(amplitudes):.2f}")
        
        r_signals = np.sqrt(x ** 2 + y ** 2)
        theta_signals = np.arctan2(y, x)
        theta_signals = np.where(theta_signals < 0, theta_signals + 2 * np.pi, theta_signals)
        
        # Grid parameters setup
        num_r = 50
        num_theta = 120
        r_edges = np.linspace(0, self.search_radius, num_r + 1)
        theta_edges = np.linspace(0, 2 * np.pi, num_theta + 1)
        r_centers = (r_edges[:-1] + r_edges[1:]) / 2
        theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
        
        # Параметри ядра та правдоподібності
        a = 3.0  # параметр для експоненційного ядра
        logger.debug(f"Kernel parameter a: {a}")
        
        noise_mask = (states != 2)
        if not np.any(noise_mask):
            sigma = 200
            logger.debug(f"Using default sigma: {sigma} (no noise markers found)")
        else:
            noise_amplitudes = markers[noise_mask][:, 2]
            sigma = np.sum(noise_amplitudes) / len(noise_amplitudes)
            logger.debug(f"Computed sigma from noise: {sigma:.2f} (from {len(noise_amplitudes)} markers)")
        
        # Debugging check only - doesn't change the value
        if sigma < 1.0:
            logger.warning(f"Sigma is very small ({sigma:.6f}), which may cause numerical issues")
        
        I_k = (self.object_amplitude_range[0] + self.object_amplitude_range[1]) / 2
        logger.debug(f"Expected object amplitude I_k: {I_k}")
        
        z_grid = np.zeros((num_r, num_theta))
        R_okil = 4
        
        # ----- Обчислення z_grid ------
        logger.debug(f"Computing z_grid with neighborhood radius R_okil = {R_okil}")
        
        # Count cells with signals for debugging
        cells_with_signals = 0
        max_signals_per_cell = 0
        
        for i in range(num_r):
            for j in range(num_theta):
                r_c = r_centers[i]
                theta_c = theta_centers[j]
                
                # Calculate distances
                d = np.sqrt(
                    r_c ** 2
                    + r_signals ** 2
                    - 2 * r_c * r_signals * np.cos(theta_c - theta_signals)
                )
                
                # Select signals within neighborhood
                mask = d <= R_okil
                n_ij = np.count_nonzero(mask)
                
                if n_ij == 0:
                    continue
                    
                cells_with_signals += 1
                max_signals_per_cell = max(max_signals_per_cell, n_ij)
                
                # Calculate average amplitude
                A_mask = amplitudes[mask]
                A_avg = A_mask.mean()
                
                # Compute kernel m_k
                m_k = np.exp(-a * d[mask] ** 2 / (num_r * num_theta))
                
                # Compute z_ij
                z_value = (1 / np.sqrt(n_ij)) * np.sum((A_mask - A_avg) * m_k)
                z_grid[i, j] = z_value
        
        # Detailed statistics of z_grid for debugging
        z_nonzero = z_grid[z_grid != 0]
        logger.debug(f"z_grid statistics: {cells_with_signals}/{num_r*num_theta} cells have signals")
        logger.debug(f"Maximum number of signals in a cell: {max_signals_per_cell}")
        
        if len(z_nonzero) > 0:
            logger.debug(f"z_grid non-zero values - min: {np.min(z_nonzero):.2f}, max: {np.max(z_nonzero):.2f}, mean: {np.mean(z_nonzero):.2f}")
        else:
            logger.warning("z_grid has no non-zero values!")
        
        # ----- Обчислення ваг ------
        logger.debug(f"Computing weights with sigma={sigma:.2f}, I_k={I_k:.2f}")
        
        marker_weights = np.zeros(len(markers_state2))
        
        # Track extreme values that may cause overflow
        max_exponent = float('-inf')
        min_exponent = float('inf')
        max_mu_ij = float('-inf')
        max_z_diff = float('-inf')
        
        # Original weight calculation with added logging
        for idx in range(len(markers_state2)):
            r = r_signals[idx]
            theta = theta_signals[idx]
            
            # Initialize L
            L = 1.0
            
            for i in range(num_r):
                for j in range(num_theta):
                    if z_grid[i, j] == 0:
                        continue
                        
                    r_center = r_centers[i]
                    theta_center = theta_centers[j]
                    
                    dist = np.sqrt((r - r_center) ** 2 + (theta - theta_center) ** 2)
                    h_ij = np.exp(-a * dist ** 2 / (num_r * num_theta))
                    
                    if h_ij < 1e-10:
                        continue
                        
                    mu_ij = I_k * h_ij
                    
                    # This is where overflow happens
                    exponent = (mu_ij * (mu_ij - 2 * z_grid[i, j])) / (2 * sigma ** 2)
                    
                    # Track extreme values for logging only
                    max_exponent = max(max_exponent, exponent)
                    min_exponent = min(min_exponent, exponent)
                    max_mu_ij = max(max_mu_ij, mu_ij)
                    max_z_diff = max(max_z_diff, abs(mu_ij - 2 * z_grid[i, j]))
                    
                    # Check if exponent is too large (for logging only)
                    if exponent > 700:  # np.exp(710) is roughly the limit before overflow
                        logger.warning(f"Potential overflow detected! Exponent = {exponent:.2f}")
                        logger.warning(f"Components: mu_ij={mu_ij:.2f}, z_grid[{i},{j}]={z_grid[i,j]:.2f}, sigma²={sigma**2:.2f}")
                        # Don't break or modify L, just log the warning
                    
                    # Original computation - no changes
                    L *= np.exp(exponent)
            
            marker_weights[idx] = L
        
        # Log statistics about weight calculation
        logger.debug(f"Exponent range: [{min_exponent:.2f}, {max_exponent:.2f}]")
        logger.debug(f"Max mu_ij: {max_mu_ij:.2f}, Max |mu_ij - 2*z_grid|: {max_z_diff:.2f}")
        logger.debug(f"Sigma squared: {sigma**2:.2f}")
        
        # Check if we have any infinite weights (for logging only)
        inf_weights = np.isinf(marker_weights)
        if np.any(inf_weights):
            logger.warning(f"{np.sum(inf_weights)} markers have infinite weights")
        
        # Original normalization
        total_weight = np.sum(marker_weights)
        if total_weight > 0:
            marker_weights /= total_weight
            logger.debug(f"Normalized weights - sum: {np.sum(marker_weights):.6f}")
        else:
            logger.warning(f"Total weight is {total_weight}, using uniform weights")
            marker_weights = np.ones(len(marker_weights)) / len(marker_weights)
        
        # Final weight statistics
        if len(marker_weights) > 0:
            min_w = np.min(marker_weights)
            max_w = np.max(marker_weights)
            entropy = -np.sum(marker_weights * np.log(marker_weights + 1e-10))
            logger.info(f"Min weight: {min_w:.8f}, Max weight: {max_w:.8f}")
            logger.info(f"Weight entropy: {entropy:.4f}")
        
        return marker_weights



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
            
            x_vals, y_vals = markers[state2_mask, 0], markers[state2_mask, 1]
            r_vals = np.sqrt(x_vals ** 2 + y_vals ** 2)
            theta_vals = np.arctan2(y_vals, x_vals)
            
            logger.debug(f"Marker positions - x range: [{np.min(x_vals):.4f}, {np.max(x_vals):.4f}], " 
                        f"y range: [{np.min(y_vals):.4f}, {np.max(y_vals):.4f}]")
            logger.debug(f"Marker positions (polar) - r range: [{np.min(r_vals):.4f}, {np.max(r_vals):.4f}], " 
                        f"θ range: [{np.min(theta_vals):.4f}, {np.max(theta_vals):.4f}]")
            
            current_center_r = np.sqrt(current_center[0]**2 + current_center[1]**2)
            current_center_theta = np.arctan2(current_center[1], current_center[0])
            
            logger.debug(f"Current center (cartesian): [{current_center[0]:.4f}, {current_center[1]:.4f}]")
            logger.debug(f"Current center (polar): [r={current_center_r:.4f}, θ={current_center_theta:.4f} rad]")
            
            if center_prev is not None:
                logger.debug(f"Previous center (cartesian): [{center_prev[0]:.4f}, {center_prev[1]:.4f}]")
                
                # Only for logging - calculate polar coordinates of previous center
                center_prev_r = np.sqrt(center_prev[0]**2 + center_prev[1]**2)
                center_prev_theta = np.arctan2(center_prev[1], center_prev[0])
                logger.debug(f"Previous center (polar): [r={center_prev_r:.4f}, θ={center_prev_theta:.4f} rad]")
            
            if center_prev2 is not None:
                logger.debug("Using 3-point velocity estimation formula")
                velocity = (3 * current_center - 4 * center_prev + center_prev2) / (2 * dt)
                logger.debug(f"3-point estimated velocity: {velocity}")
            elif t == 0:
                logger.debug("First iteration, velocity set to 0")
                velocity = 0
            else:
                logger.debug("Using 2-point velocity estimation with mixed coordinates")
                
                # ISSUE: Mixing coordinate systems in original code
                velocity_r = (current_center_r - center_prev[0]) / dt
                velocity_theta = (current_center_theta - center_prev[1]) / dt
                
                logger.error(f"COORDINATE MIXING ERROR: Calculating velocity_r = ({current_center_r:.4f} - {center_prev[0]:.4f}) / {dt}")
                logger.error(f"COORDINATE MIXING ERROR: Calculating velocity_theta = ({current_center_theta:.4f} - {center_prev[1]:.4f}) / {dt}")
                
                velocity = np.array([velocity_r, velocity_theta])
                logger.debug(f"Resulting velocity vector: [{velocity_r:.4f}, {velocity_theta:.4f}]")
                
                # Only for logging - calculate what would be the correct velocity in polar coordinates
                if center_prev is not None:
                    correct_velocity_r = (current_center_r - center_prev_r) / dt
                    correct_velocity_theta = (current_center_theta - center_prev_theta) / dt
                    logger.debug(f"For reference - correct polar velocity would be: [{correct_velocity_r:.4f}, {correct_velocity_theta:.4f}]")

            noise = np.random.normal(0, self.noise_std, size=(np.sum(state2_mask), 2))
            logger.debug(f"Generated noise with mean: {np.mean(noise):.6f}, std: {np.std(noise):.6f}")
            
            markers[state2_mask, :2] += noise * dt
            logger.debug(f"Added noise to markers")
            
            markers[state2_mask, :2] = self.clip_radius(markers[state2_mask, :2], 10)
            logger.debug(f"Clipped markers to radius 10")
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
        """Головний цикл детекції"""
        logger.info(f"Starting detection process with {num_iterations} iterations")
        
        markers, states, object_indices = self.initialize_markers()
        logger.info(f"Initialized {len(markers)} markers: {np.sum(states==1)} in state 1, {np.sum(states==2)} in state 2")
        logger.debug(f"Identified {len(object_indices)} object indices")

        for t in range(num_iterations):
            logger.info(f"\n{'='*20} ITERATION {t} {'='*20}")
            
            # Генерація нових маркерів стану 1
            logger.debug(f"Current marker count before generation: {len(markers)}")
            new_markers, new_states = self.generate_new_markers(len(markers))
            
            if len(new_markers) > 0:
                logger.debug(f"Adding {len(new_markers)} new markers with state 1")
                markers = np.vstack((markers, new_markers))
                states = np.hstack((states, new_states))
                logger.debug(f"Updated marker count: {len(markers)}")
            else:
                logger.debug("No new markers generated")
                
            # Візуалізація початкового стану
            if(t==0):
                logger.debug("Visualizing initial state")
                self.visualize(markers, states, t)
                
            # Оновлення станів маркерів
            logger.debug("Updating marker states")
            states_before = np.copy(states)
            states = self.update_states(states)
            
            state_changes = np.sum(states != states_before)
            logger.debug(f"Changed states for {state_changes} markers ({state_changes/len(states)*100:.2f}%)")
            logger.debug(f"Current marker distribution: {np.sum(states==1)} in state 1, {np.sum(states==2)} in state 2")
            
            # Debugging checkpoint
            if(t == 1):
                logger.debug("Special checkpoint at t=1")
                pass
                
            # Обчислення ваг за допомогою нової функції
            logger.debug("Computing weights for markers")
            weights = self.compute_weights(markers, states)
            
            if len(weights) > 0:
                logger.debug(f"Weight statistics - min: {np.min(weights):.8f}, max: {np.max(weights):.8f}, sum: {np.sum(weights):.8f}")
            else:
                logger.warning("No weights returned from compute_weights")
                
            # Обчислення позиції центру мас
            logger.debug("Computing position (center of mass)")
            self.current_center, P = self.compute_position_gaussian(markers, states, weights, t)
            
            if self.current_center is not None:
                logger.info(f"Current center position: [{self.current_center[0]:.4f}, {self.current_center[1]:.4f}], confidence P={P:.4f}")
                
                # Log comparison with previous center if available
                if self.center_prev is not None:
                    distance = np.linalg.norm(self.current_center - self.center_prev)
                    logger.debug(f"Movement since last iteration: {distance:.4f} units")
            else:
                logger.warning("Failed to compute center position")
                
            # Ресемплінг маркерів стану 2
            logger.debug("Performing resampling of state 2 markers")
            markers_before = len(markers)
            markers, weights = self.resample_step(markers, weights, states)
            logger.debug(f"After resampling: {len(markers)} markers (change: {len(markers) - markers_before})")
            
            # Закоментовані виклики руху - відображаємо це в логах
            logger.debug("Note: move_noise_markers and move_object_signals are commented out in the original code")
            # markers = self.move_noise_markers(markers, dt,dt)
            # markers = self.move_object_signals(markers, object_indices, angular_speed=0.1, dt=dt)

            # Оновлення позицій маркерів
            logger.debug("Updating marker positions")
            markers = self.update_markers(markers, states, dt, self.current_center, self.center_prev, t)
            
            # Збереження поточного центру як попереднього для наступної ітерації
            logger.debug("Saving current center as previous center for next iteration")
            self.center_prev = self.current_center
            
            # Візуалізація за потреби
            if t % 1 == 0 or P >= 0.9:
                logger.debug(f"Visualizing at iteration {t} (P={P:.4f})")
                self.visualize(markers, states, t, self.current_center)
                
            # Видалення шумових маркерів
            logger.debug(f"Marker distribution before filtering: {np.sum(states==1)} in state 1, {np.sum(states==2)} in state 2")
            state2_mask = states == 2
            markers_before = len(markers)
            
            # CRITICAL ISSUE: Removing all state 1 markers
            markers = markers[state2_mask]
            states = states[state2_mask]
            
            removed = markers_before - len(markers)
            logger.warning(f"ISSUE DETECTED: Removed all state 1 markers! Removed {removed} markers, kept {len(markers)} state 2 markers")
            logger.debug(f"Marker distribution after filtering: {np.sum(states==1)} in state 1, {np.sum(states==2)} in state 2")
            
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
