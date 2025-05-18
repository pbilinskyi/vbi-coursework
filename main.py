import numpy as np
import matplotlib.pyplot as plt


class ObjectDetector:
    def __init__(self, search_radius=10, N0=100, M=200, noise_std=0.1,
                 prob_change_state=0.1,object_pos = (5, 1.75*np.pi),
                 object_amplitude_range = (1000,1500), num_object_signals = 20, amplitude_range=(200,1200)):
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

    def initialize_markers(self):
        """Генерація початкових маркерів (стан 2) з рівномірним розподілом по круговій області."""
        # np.random.seed(41)

        signals_radii = self.search_radius * np.sqrt(np.random.uniform(0, 1, self.N0))
        signals_angles = np.random.uniform(0, 2 * np.pi, self.N0)
        signals_amplitudes = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], self.N0)

        object_signals_radii = np.random.normal(self.object_pos_radii, 0.1, self.num_object_signals)
        object_signals_angles = np.random.normal(self.object_pos_angle, 0.1, self.num_object_signals)
        object_signals_amplitudes = np.random.uniform(self.object_amplitude_range[0], self.object_amplitude_range[1],
                                                      self.num_object_signals)

        radii = np.concatenate((signals_radii, object_signals_radii))
        angles = np.concatenate((signals_angles, object_signals_angles))
        amplitudes = np.concatenate((signals_amplitudes, object_signals_amplitudes))

        markers = np.column_stack((
            radii * np.cos(angles),
            radii * np.sin(angles),
            amplitudes
        ))

        states = np.full(self.N0 + self.num_object_signals, 2)

        #  індекси маркерів, що відповідають об'єктам
        object_indices = np.arange(self.N0, self.N0 + self.num_object_signals)

        return markers, states, object_indices

    def generate_new_markers(self, Nk_prev):
        """Генерація нових маркерів (стан 1). Для них амплітуда із діапазону шуму."""
        num_new = self.M - Nk_prev
        if num_new <= 0:
            return np.array([]), np.array([])
        angles = np.random.uniform(0, 2 * np.pi, num_new)
        radii = self.search_radius * np.sqrt(np.random.uniform(0, 1, num_new))
        # Для маркерів стану 1 використовуємо амплітуди з діапазону, заданого в amplitude_range)
        amplitudes = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], num_new)
        new_markers = np.column_stack((
            radii * np.cos(angles),
            radii * np.sin(angles),
            amplitudes
        ))
        self.sigma = np.sum(amplitudes) / len(amplitudes)
        return new_markers, np.ones(num_new)  # стан 1 позначається як 1

    def update_states(self, states):
        """Оновлення станів маркерів: з ймовірністю prob_change_state маркер змінює стан."""
        state_changes = np.random.random(len(states)) < self.prob_change_state
        states[state_changes] = 3 - states[state_changes]  # якщо був 1, стане 2; якщо був 2, стане 1
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
        # Використовуємо лише маркери стану 2
        state2_mask = (states == 2)
        if not np.any(state2_mask):
            return np.array([])
        markers_state2 = markers[state2_mask]

        # Перетворюємо координати в полярні
        x = markers_state2[:, 0]
        y = markers_state2[:, 1]
        amplitudes = markers_state2[:, 2]
        r_signals = np.sqrt(x ** 2 + y ** 2)
        theta_signals = np.arctan2(y, x)
        theta_signals = np.where(theta_signals < 0, theta_signals + 2 * np.pi, theta_signals)

        # Параметри сітки
        num_r = 50
        num_theta = 120
        r_edges = np.linspace(0, self.search_radius, num_r + 1)
        theta_edges = np.linspace(0, 2 * np.pi, num_theta + 1)
        r_centers = (r_edges[:-1] + r_edges[1:]) / 2
        theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2

        # Параметри ядра та правдоподібності
        a = 3.0  # параметр для експоненційного ядра
        noise_mask = (states != 2)
        if not np.any(noise_mask):  # Якщо шумових маркерів немає, ставимо стандартне значення sigma
            sigma = 200
        else:
            noise_amplitudes = markers[noise_mask][:, 2]
            sigma = np.sum(noise_amplitudes) / len(noise_amplitudes)

        I_k = (self.object_amplitude_range[0] + self.object_amplitude_range[1]) / 2

        z_grid = np.zeros((num_r, num_theta))
        R_okil = 4

        def delta_theta(theta1, theta2):
            d = np.abs(theta1 - theta2)
            return np.minimum(d, 2 * np.pi - d)

        # Проходимо по кожній клітинці
        for i in range(num_r):
            for j in range(num_theta):
                r_c = r_centers[i]
                theta_c = theta_centers[j]

                # 1) обчислюємо відстань до всіх сигналів через косинус
                #    d_k = sqrt(r_c^2 + r_k^2 - 2 * r_c * r_k * cos(theta_c - theta_k))
                d = np.sqrt(
                    r_c ** 2
                    + r_signals ** 2
                    - 2 * r_c * r_signals * np.cos(theta_c - theta_signals)
                )

                # 2) відбираємо ті, що в колі радіуса R_okil
                mask = d <= R_okil
                n_ij = np.count_nonzero(mask)
                if n_ij == 0:
                    continue

                # 3) середнє амплітуд і відхилення
                A_mask = amplitudes[mask]
                A_avg = A_mask.mean()

                # 4) ядро m_k
                m_k = np.exp(-a * d[mask] ** 2 / (num_r * num_theta))

                # 5) власне z_ij
                z_grid[i, j] = (1 / np.sqrt(n_ij)) * np.sum((A_mask - A_avg) * m_k)
        self.draw_z_grid(theta_edges, r_edges, z_grid)

        marker_weights = np.zeros(len(markers_state2))
        for idx in range(len(markers_state2)):
            r = r_signals[idx]
            theta = theta_signals[idx]

            L = 1.0
            for i in range(num_r):
                for j in range(num_theta):
                    r_center = r_centers[i]
                    theta_center = theta_centers[j]

                    dist = np.sqrt((r - r_center) ** 2 + (theta - theta_center) ** 2)
                    h_ij = np.exp(-a * dist ** 2 / (num_r * num_theta))

                    if h_ij < 1e-10:
                        continue
                    mu_ij = I_k * h_ij
                    exponent = (mu_ij * (mu_ij - 2 * z_grid[i, j])) / (2 * sigma ** 2)

                    L *= np.exp(exponent)

            marker_weights[idx] = L

        total_weight = np.sum(marker_weights)
        if total_weight > 0:
            marker_weights /= total_weight
        else:
            marker_weights = np.ones(len(marker_weights)) / len(marker_weights)
        if len(marker_weights) > 0:
            print(f"Min weight: {np.min(marker_weights):.8f}, Max weight: {np.max(marker_weights):.8f}")
            print(f"Weight entropy: {-np.sum(marker_weights * np.log(marker_weights + 1e-10)):.4f}")
        # if len(marker_weights) > 0:
        #     max_weight_idx = np.argmax(marker_weights)
        #     max_weight_marker = markers_state2[max_weight_idx]
        #     max_weight_r = r_signals[max_weight_idx]
        #     max_weight_theta = theta_signals[max_weight_idx]
        #
        #     # Переводимо полярні координати назад у декартові для виведення
        #     max_weight_x = max_weight_marker[0]
        #     max_weight_y = max_weight_marker[1]
        #
        #     print(f"Позиція маркера з найбільшою вагою:")
        #     print(
        #         f"  Полярні координати: r = {max_weight_r:.2f}, θ = {max_weight_theta:.2f} рад (≈ {np.degrees(max_weight_theta):.2f}°)")
        #     print(f"  Декартові координати: x = {max_weight_x:.2f}, y = {max_weight_y:.2f}")
        #     print(f"  Вага: {marker_weights[max_weight_idx]:.6f}")

        return marker_weights
    def compute_position_gaussian(self, markers, states, weights, t, sigma = 0.5):
        """ Обчислення позиції центру мас"""

        state2_mask = states == 2
        markers_state2 = markers[state2_mask]

        # тільки координати x,y (без амплітуди)
        positions = markers_state2[:, :2]

        # Обчислюємо зважений центр мас
        x_k = np.sum(weights[:, np.newaxis] * positions, axis=0)

        distances = np.sum((positions - x_k) ** 2, axis=1)

        delta_approx = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-distances / (2 * sigma ** 2))

        # Сумуємо всі значення для отримання P
        P = np.sum(delta_approx)

        x, y = x_k
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        if theta < 0:
            theta += 2 * np.pi
        x_k_polar = (r,theta)
        P_max = len(positions) * (1.0 / (sigma * np.sqrt(2*np.pi)))

        if(P/P_max>0.9):
            pass
        print(f"\nP= {P}\n")

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
        indices_state_2 = np.where(states == 2)[0]
        markers_state_2 = markers[indices_state_2]

        new_markers_state_2 = self.multinomial_resampling(markers_state_2, weights)
        # Оновлюємо ваги (після ресемплінгу вони рівномірні)
        new_weights_state_2 = np.ones(len(new_markers_state_2)) / len(new_markers_state_2)

        new_markers = markers.copy()
        new_markers[indices_state_2] = new_markers_state_2

        return new_markers, new_weights_state_2
    def clip_radius(self, markers, max_radius):
        r = np.linalg.norm(markers, axis=1)

        mask = r > max_radius
        if np.any(mask):
            markers[mask] *= max_radius / r[mask, np.newaxis]

        return markers

    def update_markers(self, markers, states, dt, current_center, center_prev, t, center_prev2=None):
        state2_mask = (states == 2)

        if np.any(state2_mask):
            x_vals, y_vals = markers[state2_mask, 0], markers[state2_mask, 1]
            r_vals = np.sqrt(x_vals ** 2 + y_vals ** 2)
            theta_vals = np.arctan2(y_vals, x_vals)
            current_center_r = np.sqrt(current_center[0]**2 + current_center[1]**2)
            current_center_theta = np.arctan2(current_center[1], current_center[0])

            if center_prev2 is not None:
                velocity = (3 * current_center - 4 * center_prev + center_prev2) / (2 * dt)
            elif t==0:
                # current_center = np.mean(markers[state2_mask, 2], axis=0)
                velocity = 0
            else:
                velocity_r = (current_center_r - center_prev[0]) / dt
                velocity_theta = (current_center_theta - center_prev[1]) / dt
                velocity = np.array([velocity_r, velocity_theta])

            noise = np.random.normal(0, self.noise_std, size=(np.sum(state2_mask), 2))


            markers[state2_mask, :2] += noise * dt

            markers[state2_mask, :2] = self.clip_radius(markers[state2_mask, :2], 10)

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
        markers, states, object_indices = self.initialize_markers()


        for t in range(num_iterations):
            # Генерація нових маркерів стану 1
            new_markers, new_states = self.generate_new_markers(len(markers))
            if len(new_markers) > 0:
                markers = np.vstack((markers, new_markers))
                states = np.hstack((states, new_states))
            if(t==0):
                self.visualize(markers, states, t)
            # Оновлення станів маркерів
            states = self.update_states(states)
            # Обчислення ваг за допомогою нової функції
            #4
            if(t == 1):
                pass
            weights = self.compute_weights(markers, states)
            #5
            self.current_center, P = self.compute_position_gaussian(markers, states, weights,t)
            #6 Ресемплінг маркерів стану 2
            markers, weights = self.resample_step(markers, weights, states)
            # markers = self.move_noise_markers(markers, dt,dt)
            # markers = self.move_object_signals(markers, object_indices, angular_speed=0.1, dt=dt)

            # Оновлення позицій маркерів
            markers = self.update_markers(markers, states, dt, self.current_center, self.center_prev,t)
            self.center_prev = self.current_center
            if t % 1 == 0 or P >=0.9:
                self.visualize(markers, states, t, self.current_center)
            # Видалення шумових маркерів для їх повторного створення на новій ітерації
            state2_mask = states == 2
            markers = markers[state2_mask]
            states = states[state2_mask]



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
