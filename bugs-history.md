History of bugs and their fixes

# Session 1 - Аналіз логів алгоритму виявлення об'єктів

На основі логів виявлено кілька критичних проблем, які перешкоджають правильній роботі алгоритму:

## 1. Проблема з обчисленням ваг (NaN значення)

```
RuntimeWarning: overflow encountered in scalar multiply
  L *= np.exp(exponent)
RuntimeWarning: invalid value encountered in divide
  marker_weights /= total_weight
```

**Причина**: При обчисленні ваг відбувається переповнення при множенні на експоненту. Значення `exponent` в формулі `L *= np.exp(exponent)` стає надто великим, що призводить до переповнення і появи NaN.

**Наслідок**: Усі ваги маркерів стають NaN, що робить неможливим коректне обчислення центру мас та подальшу роботу алгоритму.

## 2. Проблема з обчисленням центру мас

```
DEBUG - Computed center of mass (cartesian): [nan, nan]
DEBUG - Computed center of mass (polar): [r=nan, θ=nan rad ≈ nan°]
DEBUG - Confidence metric P: nan
```

**Наслідок**: Оскільки ваги стали NaN, центр мас також обчислюється як NaN, що робить неможливим відстеження руху об'єкта.

## 3. Проблема з видаленням маркерів стану 1

```
WARNING - ISSUE DETECTED: Removed all state 1 markers! Removed 489 markers, kept 511 state 2 markers
DEBUG - Marker distribution after filtering: 0 in state 1, 511 in state 2
```

**Проблема**: Наприкінці кожної ітерації всі маркери стану 1 видаляються. Це суперечить теоретичному алгоритму Васіна, який передбачає збереження обох типів маркерів.

**Наслідок**: Алгоритм втрачає частину маркерів, що необхідні для правильного дослідження простору пошуку. Це порушує баланс між дослідженням нових областей і використанням наявних знань.

## 4. Змішування координатних систем (підтверджено логами)

```
ERROR - COORDINATE MIXING ERROR: Calculating velocity_r = (nan - nan) / 1
ERROR - COORDINATE MIXING ERROR: Calculating velocity_theta = (nan - nan) / 1
```

Хоча значення тут NaN через попередні помилки, логи підтверджують проблему змішування координатних систем, яку ми виявили раніше.

## Рекомендації для виправлення:

1. **Виправлення обчислення ваг**:
   - Обмежити значення `exponent` перед обчисленням експоненти
   - Використовувати логарифмічні операції для запобігання переповнення
   - Додати перевірку на переповнення і встановлювати граничні значення

   ```python
   # Приклад:
   exponent = np.clip(exponent, -100, 100)  # Обмежуємо діапазон exponent
   L *= np.exp(exponent)
   ```

2. **Виправлення видалення маркерів**:
   - Зберігати обидва типи маркерів (стану 1 і 2) протягом ітерацій
   - Обмежувати загальну кількість маркерів, видаляючи випадкову підмножину, якщо потрібно

   ```python
   # Замінити:
   state2_mask = states == 2
   markers = markers[state2_mask]
   states = states[state2_mask]
   
   # На:
   if len(markers) > self.M:
       # Зберігаємо всі маркери стану 2
       indices_state2 = np.where(states == 2)[0]
       # Випадково вибираємо маркери стану 1 до досягнення ліміту
       indices_state1 = np.where(states == 1)[0]
       keep_state1 = np.random.choice(indices_state1, min(len(indices_state1), self.M - len(indices_state2)), replace=False)
       keep_indices = np.concatenate((indices_state2, keep_state1))
       markers = markers[keep_indices]
       states = states[keep_indices]
   ```

3. **Виправлення змішування координат**:
   - Забезпечити узгодження координатних систем при обчисленні швидкості

Ці проблеми взаємопов'язані: NaN при обчисленні ваг призводить до NaN у центрі мас, що призводить до подальших проблем. Виправлення проблеми переповнення при обчисленні ваг є найбільш критичним першим кроком.
