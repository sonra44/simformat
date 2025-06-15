import numpy as np
import logging
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Config:
    """
    Класс Config хранит все конфигурационные параметры симуляции.
    Все параметры строго типизированы и должны соответствовать физической реальности.
    
    Используется dataclass с frozen=True для обеспечения неизменяемости
    и предотвращения случайных изменений параметров во время выполнения.
    """
    
    # --- Настройки логирования ---
    LOG_LEVEL: int = logging.INFO  # Уровень логирования по умолчанию
    
    # --- Общие настройки симуляции ---
    TIME_STEP: float = 0.5  # [секунды] Временной шаг симуляции
    MAX_TIME: float = 3600.0  # [секунды] Максимальное время симуляции (1 час)
    REAL_TIME_MODE: bool = True  # Включить режим реального времени
    SIMULATION_RATE: float = 60.0  # [Гц] Частота обновления симуляции
    SIMULATION_FRAME_TIME: float = 1.0 / 60.0  # [с] Фиксированный шаг времени для стабильности физики

    # --- Физические параметры бота ---
    BOT_MASS_INITIAL: float = 10.0  # [кг] Начальная масса бота (без компонентов)
    BOT_RADIUS: float = 0.5  # [м] Физический радиус бота
    GRAVITY: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81], dtype=float))  # [м/с^2] Вектор гравитации

    # --- Управление и движение ---
    MAX_THRUST_N: float = 100.0  # [Н] Максимальная сила тяги
    MAX_TORQUE_NM: float = 50.0  # [Н*м] Максимальный крутящий момент
    MAX_ACCELERATION: float = 5.0  # [м/с²] Максимальное ускорение

    # --- Границы среды ---
    BOUNDS: dict = field(default_factory=lambda: {"x": (-100.0, 100.0), "y": (-100.0, 100.0), "z": (-100.0, 100.0)}) # [метры] Границы симуляции

    # --- Визуализация ---
    PLOT_HISTORY: int = 500  # Количество точек для отображения на графиках Matplotlib
    TELEMETRY_HISTORY_LENGTH: int = 1000  # Количество точек для хранения в истории телеметрии
    UPDATE_RATE_VISUALIZER_FPS: float = 30.0  # [кадры/сек] Частота обновления графического визуализатора
    ASCII_VISUALIZER_ENABLED: bool = True  # Включаем ASCII визуализацию
    UPDATE_RATE_ASCII_FPS: float = 1.0 # [кадры/сек] Частота обновления ASCII визуализатора (меньше, чем графический)
    VISUALIZATION_RATE: float = 30.0  # [FPS] Частота обновления визуализации

    # --- Пороги безопасности и состояния (Проценты и температуры) ---
    MAX_VELOCITY: float = 50.0  # [м/с] Максимальная допустимая скорость
    LOW_BATTERY_THRESHOLD: float = 20.0  # [%] Порог низкого заряда батареи
    WARNING_BATTERY_THRESHOLD: float = 30.0 # [%] Порог предупреждения о заряде батареи
    CRITICAL_INTEGRITY_THRESHOLD: float = 30.0  # [%] Порог критической целостности каркаса
    
    # Термальные пороги
    THERMAL_WARNING_THRESHOLD_C: float = 60.0  # [°C] Порог предупреждения о температуре
    OVERHEAT_THRESHOLD_C: float = 80.0  # [°C] Порог критического перегрева

    OVERHEAT_THRESHOLD_C: float = 80.0  # [°C] Порог критического перегрева ядра
    WARNING_TEMPERATURE_C: float = 70.0  # [°C] Порог предупреждения о перегреве ядра
    FREEZE_THRESHOLD_C: float = -20.0  # [°C] Порог критического замерзания ядра
    WARNING_TEMPERATURE_C_LOW: float = -10.0  # [°C] Порог предупреждения о низкой температуре ядра

    OVERLOAD_THRESHOLD_W: float = 1500.0  # [Ватты] Порог перегрузки по мощности

    # --- Параметры каркаса (FrameCore) ---
    FRAME_CORE_MASS: float = 5.0  # [кг] Базовая масса каркаса
    MAX_ACCELERATION_TOLERANCE: float = 30.0  # [м/с^2] Максимальное допустимое ускорение
    MAX_ANGULAR_ACCELERATION_TOLERANCE: float = 10.0  # [рад/с^2] Максимальное допустимое угловое ускорение
    STRUCTURAL_INTEGRITY_DEGRADATION_RATE_PER_G_PER_SEC: float = 0.01  # [%/g/с] Скорость деградации целостности на единицу перегрузки в секунду

    # --- Параметры структурной целостности ---
    STRUCTURAL_STRESS_FACTOR: float = 0.001  # Коэффициент влияния линейного ускорения на стресс
    STRUCTURAL_ANGULAR_STRESS_FACTOR: float = 0.005  # Коэффициент влияния углового ускорения на стресс
    STRUCTURAL_INTEGRITY_THRESHOLD: float = 0.7  # Порог стресса для начала деградации целостности
    STRUCTURAL_DEGRADATION_RATE: float = 0.1  # Скорость деградации целостности при превышении порога

    # --- Параметры энергетической системы (PowerSystems) ---
    MAIN_BATTERY_CAPACITY_WH: float = 2000.0  # [Ватт-часы] Емкость основной батареи
    RESERVE_BATTERY_CAPACITY_WH: float = 500.0  # [Ватт-часы] Емкость резервной батареи
    POWER_CONSUMPTION_IDLE_W: float = 20.0  # [Вт] Потребление энергии в режиме ожидания
    POWER_CONSUMPTION_ACTIVE_W: float = 100.0  # [Вт] Потребление энергии в активном режиме
    POWER_CONSUMPTION_PEAK_W: float = 500.0  # [Вт] Пиковое потребление энергии
    INITIAL_CHARGE_PERCENTAGE: float = 100.0  # [%] Начальный заряд батарей
    MAX_CHARGE_RATE_W: float = 200.0  # [Ватты] Максимальная скорость заряда
    MAX_DISCHARGE_RATE_W: float = 600.0  # [Ватты] Максимальная скорость разряда (пиковая)
    
    # --- Параметры солнечных панелей ---
    SOLAR_PANEL_AREA_M2: float = 4.0  # [м²] Площадь солнечных панелей
    SOLAR_PANEL_EFFICIENCY: float = 0.22  # [%] КПД солнечных панелей (22% - современные космические панели)
    MIN_SOLAR_IRRADIANCE: float = 100.0  # [Вт/м²] Минимальная солнечная радиация для генерации
    MAX_SOLAR_IRRADIANCE: float = 1361.0  # [Вт/м²] Максимальная солнечная радиация (солнечная постоянная)

    # --- Параметры массы компонентов энергетической системы ---
    MAIN_BATTERY_MASS_KG: float = 15.0  # [кг] Масса основной батареи
    RESERVE_BATTERY_MASS_KG: float = 5.0  # [кг] Масса резервной батареи
    SOLAR_PANEL_MASS_KG: float = 3.0  # [кг] Масса солнечных панелей
    POWER_SYSTEM_BASE_MASS_KG: float = 2.0  # [кг] Масса базовых компонентов энергосистемы
    
    # --- Параметры термической системы (ThermalSystem) ---
    STEFAN_BOLTZMANN_CONSTANT: float = 5.67e-8  # [Вт/(м²·К⁴)] Постоянная Стефана-Больцмана
    HEAT_TRANSFER_COEFFICIENT_CORE_RADIATOR: float = 10.0  # [Вт/(м²·К)] Коэффициент теплопередачи между ядром и радиатором
    RADIATOR_SURFACE_AREA_M2: float = 1.5  # [м²] Площадь поверхности радиатора
    CORE_THERMAL_MASS_J_PER_C: float = 5000.0  # [Дж/°C] Теплоёмкость ядра
    RADIATOR_THERMAL_MASS_J_PER_C: float = 2000.0  # [Дж/°C] Теплоёмкость радиатора
    EMISSIVITY: float = 0.95  # Коэффициент излучения (чернота) радиатора
    INITIAL_CORE_TEMPERATURE_C: float = 20.0  # [°C] Начальная температура ядра
    INITIAL_RADIATOR_TEMPERATURE_C: float = 10.0  # [°C] Начальная температура радиатора
    THERMAL_SYSTEM_MASS_KG: float = 8.0  # [кг] Общая масса термической системы
    THERMAL_CONDUCTIVITY: float = 50.0  # [Вт/(м·К)] Теплопроводность между ядром и радиатором
    CONTACT_AREA_M2: float = 0.1  # [м^2] Площадь контакта между ядром и радиатором
    AMBIENT_TEMPERATURE_C: float = -270.0  # [°C] Температура окружающей среды (космос)
    SPACE_TEMPERATURE_K: float = 2.7  # [K] Температура космического пространства (космический микроволновый фон)

    # --- Термальная система ---
    INITIAL_CORE_TEMPERATURE_C: float = 20.0  # [°C] Начальная температура ядра
    INITIAL_RADIATOR_TEMPERATURE_C: float = 10.0  # [°C] Начальная температура радиатора
    THERMAL_SYSTEM_BASE_POWER_W: float = 5.0  # [Вт] Базовое энергопотребление термальной системы
    MIN_CORE_TEMPERATURE_C: float = 10.0  # [°C] Минимальная температура ядра
    MIN_RADIATOR_TEMPERATURE_C: float = -50.0  # [°C] Минимальная температура радиатора
    AMBIENT_TEMPERATURE_C: float = 20.0  # [°C] Температура окружающей среды
    
    # Термальные параметры
    CORE_THERMAL_MASS_J_PER_C: float = 2000.0  # [Дж/°C] Теплоемкость ядра
    RADIATOR_THERMAL_MASS_J_PER_C: float = 1000.0  # [Дж/°C] Теплоемкость радиатора
    HEAT_TRANSFER_COEFFICIENT_CORE_RADIATOR: float = 50.0  # [Вт/°C] Коэффициент теплопередачи между ядром и радиатором
    RADIATOR_SURFACE_AREA_M2: float = 1.0  # [м²] Площадь поверхности радиатора
    EMISSIVITY: float = 0.9  # Коэффициент излучения радиатора
    STEFAN_BOLTZMANN_CONSTANT: float = 5.67e-8  # [Вт/(м²·K⁴)] Постоянная Стефана-Больцмана
    
    # Теплообмен с окружающей средой
    AMBIENT_HEAT_TRANSFER_COEFFICIENT: float = 10.0  # [Вт/(м²·°C)] Коэффициент теплообмена с окружающей средой

    # --- Параметры окружения (Environment) ---
    DEFAULT_SOLAR_IRRADIANCE_W_M2: float = 1000.0  # [Вт/м²] Солнечная радиация на поверхности Земли в ясный день
    ENVIRONMENT_DRAG_COEFFICIENT: float = 0.5  # Коэффициент сопротивления воздуха

    # --- Критические пороги для агента ---
    CRITICAL_BATTERY_LEVEL: float = 5.0  # [%] Критический уровень заряда батареи
    MAX_CORE_TEMPERATURE_C: float = 90.0  # [°C] Максимальная допустимая температура ядра
    CRITICAL_TEMPERATURE_C: float = 85.0  # [°C] Критическая температура ядра
    CRITICAL_FRAME_INTEGRITY: float = 20.0  # [%] Критический уровень целостности каркаса
    CRITICAL_INTEGRITY_LEVEL: float = 20.0  # [%] Критический уровень целостности (для агента)

    # --- Параметры агента ---
    AGENT_DECISION_INTERVAL: float = 0.5  # [с] Интервал между принятием решений
    AGENT_MAX_LINEAR_FORCE: float = 1000.0  # [Н] Максимальная линейная сила
    AGENT_MAX_TORQUE: float = 100.0  # [Н*м] Максимальный крутящий момент

    # --- Параметры потребления энергии ---
    POWER_CONSUMPTION_BASE_W: float = 50.0  # [Вт] Базовое энергопотребление
    POWER_CONSUMPTION_PROPULSION_ION_W: float = 200.0  # [Вт] Потребление ионных двигателей
    POWER_CONSUMPTION_COMPUTATION_W: float = 30.0  # [Вт] Потребление вычислительной системы
    POWER_CONSUMPTION_QIKOS_OVERHEAD_W: float = 10.0  # [Вт] Накладные расходы QikOS
    EFFICIENCY_LOSS_HEAT_FACTOR: float = 0.3  # 30% потерь в тепло

    # --- Энергетическая система ---
    MAIN_BATTERY_CAPACITY_WH: float = 10000.0  # [Вт*ч] Емкость основной батареи
    RESERVE_BATTERY_CAPACITY_WH: float = 2000.0  # [Вт*ч] Емкость резервной батареи
    MAIN_BATTERY_MASS_KG: float = 15.0  # [кг] Масса основной батареи
    RESERVE_BATTERY_MASS_KG: float = 5.0  # [кг] Масса резервной батареи
    SOLAR_PANEL_MASS_KG: float = 8.0  # [кг] Масса солнечных панелей
    
    # Энергетические параметры
    POWER_CONSUMPTION_COMPUTER_W: float = 50.0  # [Вт] Базовое потребление компьютера
    POWER_CONSUMPTION_SENSORS_W: float = 20.0  # [Вт] Базовое потребление сенсоров
    POWER_CONSUMPTION_IDLE_W: float = 10.0  # [Вт] Базовое потребление в режиме ожидания
    BATTERY_DISCHARGE_EFFICIENCY: float = 0.95  # [%] КПД разряда батарей
    BATTERY_CHARGE_EFFICIENCY: float = 0.90  # [%] КПД заряда батарей
    SOLAR_PANEL_EFFICIENCY: float = 0.22  # [%] КПД солнечных панелей
    SOLAR_PANEL_AREA_M2: float = 2.0  # [м²] Площадь солнечных панелей
    SOLAR_PANEL_THERMAL_FACTOR: float = 0.15  # [%] Доля энергии, уходящая в тепло
    
    # Пороги и ограничения
    OVERLOAD_THRESHOLD_W: float = 1000.0  # [Вт] Порог перегрузки системы
    LOW_BATTERY_THRESHOLD: float = 20.0  # [%] Порог низкого заряда батареи
    WARNING_BATTERY_THRESHOLD: float = 30.0  # [%] Порог предупреждения о низком заряде

    # --- Энергопотребление систем ---
    COMPUTER_BASE_POWER_W: float = 15.0  # [Вт] Базовое потребление компьютера
    SENSORS_BASE_POWER_W: float = 5.0  # [Вт] Базовое потребление сенсоров
    COMMS_BASE_POWER_W: float = 2.0  # [Вт] Базовое потребление систем связи
    THERMAL_CONTROL_BASE_POWER_W: float = 10.0  # [Вт] Базовое потребление термоконтроля
    NAVIGATION_BASE_POWER_W: float = 3.0  # [Вт] Базовое потребление навигации
    
    # Потребление при активных действиях
    THRUSTER_POWER_PER_NEWTON_W: float = 0.5  # [Вт/Н] Потребление на единицу тяги
    COOLING_POWER_PER_WATT_W: float = 0.2  # [Вт/Вт] Дополнительное потребление на активное охлаждение
    COOLING_POWER_PER_DEGREE_W: float = 2.0  # [Вт/°C] Потребление энергии на охлаждение на градус выше порога

    # --- Параметры управления ботом ---
    AGENT_SPEED_FACTOR: float = 2.0  # [м/с] Скорость движения бота
    AGENT_ROTATION_SPEED: float = 1.0  # [рад/с] Скорость вращения бота
    BOT_CONTROL_SENSITIVITY: float = 1.0  # Чувствительность управления (0.1-2.0)
    JOYSTICK_SENSITIVITY: float = 0.8  # Чувствительность виртуального джойстика
    TOUCH_DEADZONE_RADIUS: float = 0.1  # Мертвая зона для сенсорного управления
    
    # Параметры Android-совместимости
    ANDROID_NO_ROOT_MODE: bool = True  # Режим работы без root прав
    ANDROID_SAFE_INPUT_MODE: bool = True  # Безопасный режим ввода для Android
    MOBILE_FRIENDLY_UI: bool = True  # Оптимизация интерфейса для мобильных устройств

# Создаем глобальный экземпляр конфигурации
config = Config()

# Экспортируем конфигурацию для использования в других модулях
__all__ = ['Config', 'config']