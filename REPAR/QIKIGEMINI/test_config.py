import numpy as np
from config import Config
import logging

def test_config_values():
    """Проверяет, что все параметры Config имеют ожидаемые типы и базовые значения."""
    print("--- Проверка config.py ---")

    # Проверка физических параметров
    assert isinstance(config.BOT_MASS_INITIAL, float) and config.BOT_MASS_INITIAL > 0
    assert isinstance(config.GRAVITY, np.ndarray) and config.GRAVITY.shape == (3,)

    # Проверка временных параметров
    assert isinstance(config.TIME_STEP, float) and config.TIME_STEP > 0
    assert isinstance(config.MAX_TIME, float) and config.MAX_TIME > 0
    assert isinstance(config.REAL_TIME_MODE, bool)

    # Проверка управления
    assert isinstance(config.THRUSTER_FORCE, float) and config.THRUSTER_FORCE > 0
    assert isinstance(config.TORQUE_MAX, float) and config.TORQUE_MAX >= 0

    # Проверка границ
    assert isinstance(config.BOUNDS, dict)
    assert all(isinstance(v, tuple) and len(v) == 2 for v in config.BOUNDS.values())
    assert config.BOUNDS["x"][0] < config.BOUNDS["x"][1] # Проверка, что нижняя граница меньше верхней

    # Проверка визуализации
    assert isinstance(config.PLOT_HISTORY, int) and config.PLOT_HISTORY > 0
    assert isinstance(config.UPDATE_RATE_VISUALIZER_FPS, float) and config.UPDATE_RATE_VISUALIZER_FPS > 0
    assert isinstance(config.ASCII_VISUALIZER_ENABLED, bool)
    assert isinstance(config.UPDATE_RATE_ASCII_FPS, float) and config.UPDATE_RATE_ASCII_FPS > 0

    # Проверка порогов безопасности
    assert isinstance(config.MAX_VELOCITY, float) and config.MAX_VELOCITY > 0
    assert isinstance(config.LOW_BATTERY_THRESHOLD, float) and 0 <= config.LOW_BATTERY_THRESHOLD <= 100
    assert config.LOW_BATTERY_THRESHOLD < config.WARNING_BATTERY_THRESHOLD
    assert isinstance(config.CRITICAL_INTEGRITY_THRESHOLD, float) and 0 <= config.CRITICAL_INTEGRITY_THRESHOLD <= 100
    assert isinstance(config.OVERHEAT_THRESHOLD_C, float)
    assert config.WARNING_TEMPERATURE_C < config.OVERHEAT_THRESHOLD_C
    assert config.FREEZE_THRESHOLD_C < config.WARNING_TEMPERATURE_C_LOW
    assert isinstance(config.OVERLOAD_THRESHOLD_W, float) and config.OVERLOAD_THRESHOLD_W > 0

    # Проверка параметров каркаса
    assert isinstance(config.FRAME_BASE_MASS, float) and config.FRAME_BASE_MASS > 0
    assert isinstance(config.MAX_ACCELERATION_TOLERANCE, float) and config.MAX_ACCELERATION_TOLERANCE >= 0
    assert isinstance(config.MAX_ANGULAR_ACCELERATION_TOLERANCE, float) and config.MAX_ANGULAR_ACCELERATION_TOLERANCE >= 0
    assert isinstance(config.STRUCTURAL_INTEGRITY_DEGRADATION_RATE_PER_G_PER_SEC, float) and config.STRUCTURAL_INTEGRITY_DEGRADATION_RATE_PER_G_PER_SEC >= 0

    # Проверка энергетической системы
    assert isinstance(config.BATTERY_CAPACITY_WH, float) and config.BATTERY_CAPACITY_WH > 0
    assert isinstance(config.RESERVE_BATTERY_CAPACITY_WH, float) and config.RESERVE_BATTERY_CAPACITY_WH >= 0
    assert isinstance(config.INITIAL_CHARGE_PERCENTAGE, float) and 0 <= config.INITIAL_CHARGE_PERCENTAGE <= 100
    assert isinstance(config.MAX_CHARGE_RATE_W, float) and config.MAX_CHARGE_RATE_W >= 0
    assert isinstance(config.MAX_DISCHARGE_RATE_W, float) and config.MAX_DISCHARGE_RATE_W >= 0
    assert isinstance(config.POWER_CONSUMPTION_IDLE_W, float) and config.POWER_CONSUMPTION_IDLE_W >= 0
    assert isinstance(config.POWER_CONSUMPTION_QIKOS_OVERHEAD_W, float) and config.POWER_CONSUMPTION_QIKOS_OVERHEAD_W >= 0

    # Проверка термической системы
    assert isinstance(config.INITIAL_CORE_TEMPERATURE_C, float)
    assert isinstance(config.INITIAL_RADIATOR_TEMPERATURE_C, float)
    assert isinstance(config.CORE_THERMAL_MASS_J_PER_C, float) and config.CORE_THERMAL_MASS_J_PER_C > 0
    assert isinstance(config.RADIATOR_THERMAL_MASS_J_PER_C, float) and config.RADIATOR_THERMAL_MASS_J_PER_C > 0
    assert isinstance(config.RADIATOR_SURFACE_AREA_M2, float) and config.RADIATOR_SURFACE_AREA_M2 > 0
    assert isinstance(config.EMISSIVITY, float) and 0 <= config.EMISSIVITY <= 1
    assert isinstance(config.STEFAN_BOLTZMANN_CONSTANT, float) and config.STEFAN_BOLTZMANN_CONSTANT > 0

    # Проверка окружения
    assert isinstance(config.DEFAULT_SOLAR_IRRADIANCE_W_M2, float) and config.DEFAULT_SOLAR_IRRADIANCE_W_M2 >= 0
    assert isinstance(config.ENVIRONMENT_DRAG_COEFFICIENT, float) and config.ENVIRONMENT_DRAG_COEFFICIENT >= 0

    # Проверка логирования
    assert isinstance(config.LOG_LEVEL, int)
    assert config.LOG_LEVEL in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]

    print("Все параметры config.py успешно проверены. Типы и базовые значения корректны.")

if __name__ == "__main__":
    test_config_values()