import time
import sys
import numpy as np
import logging
import os
from collections import deque
from typing import Dict, Any, Deque, Tuple

from config import Config
from logger import Logger


class DisplayMode:
    """Определяет режимы отображения ASCII-визуализатора."""

    SIMPLIFIED = "simplified"
    DETAILED = "detailed"
    DEBUG = "debug"


class VisualizationConfig:
    """
    Класс для хранения конфигурации ASCII-визуализации.
    """

    # Размеры окна вывода (в символах)
    WIDTH: int = 80
    HEIGHT: int = 20

    # Символы для отрисовки
    BOT_CHAR: str = "B"  # Символ бота
    TARGET_CHAR: str = "T"  # Символ цели
    EMPTY_CHAR: str = "."  # Символ пустого пространства
    OBSTACLE_CHAR: str = "X"  # Символ препятствия (если будут)

    # Масштабирование координат (сколько метров на символ)
    SCALE_X: float = 5.0  # Каждый символ по X представляет 5 метров
    SCALE_Y: float = 5.0  # Каждый символ по Y представляет 5 метров
    SCALE_Z: float = 5.0  # Каждый символ по Z представляет 5 метров

    # Смещение для центрирования вида (чтобы бот был примерно в центре)
    # Эти значения будут адаптированы динамически
    OFFSET_X: float = 0.0
    OFFSET_Y: float = 0.0
    OFFSET_Z: float = 0.0

    # Период обновления визуализатора в секундах
    UPDATE_INTERVAL: float = 0.1

    # Режим отображения по умолчанию
    DEFAULT_DISPLAY_MODE: str = DisplayMode.SIMPLIFIED

    # Видимость компонентов (в зависимости от режима)
    SHOW_ACCELERATION: bool = True
    SHOW_ANGULAR_VELOCITY: bool = True
    SHOW_HEALTH: bool = True


class AsciiVisualizer:
    """
    ASCII-визуализатор для отображения состояния QIKI-бота в консоли.
    """

    def __init__(self, config: VisualizationConfig = VisualizationConfig()):
        self.logger = Logger.get_logger("ascii_visualizer")
        self.config = config
        self.last_update_time = time.time()
        self.display_mode = self.config.DEFAULT_DISPLAY_MODE
        self.last_bot_pos = None  # Для отслеживания движения

        self.logger.info(f"AsciiVisualizer initialized in {self.display_mode} mode.")

        # Очереди для хранения последних значений для трендов (если нужны)
        self.temp_history: Deque[float] = deque(maxlen=self.config.WIDTH)
        self.power_history: Deque[float] = deque(maxlen=self.config.WIDTH)
        self.integrity_history: Deque[float] = deque(maxlen=self.config.WIDTH)

    def set_display_mode(self, mode: str) -> None:
        """Устанавливает режим отображения визуализатора."""
        if mode in [DisplayMode.SIMPLIFIED, DisplayMode.DETAILED, DisplayMode.DEBUG]:
            self.display_mode = mode
            self.logger.info(f"Display mode set to {self.display_mode}.")
        else:
            self.logger.warning(f"Invalid display mode: {mode}. Keeping {self.display_mode}.")

    def _clear_console(self) -> None:
        """Очищает консоль."""
        os.system("cls" if os.name == "nt" else "clear")

    def _world_to_screen(
        self, pos: np.ndarray, bot_pos: np.ndarray
    ) -> Tuple[int, int]:
        """Преобразует мировые координаты в экранные (символьные)."""
        # Смещаем относительно бота, чтобы он был примерно в центре
        display_center_x = self.config.WIDTH // 2
        display_center_y = self.config.HEIGHT // 2

        # Рассчитываем смещение так, чтобы bot_pos была в центре экрана
        # Переворачиваем ось Y для соответствия консоли (сверху вниз)
        screen_x = int(
            display_center_x + (pos[0] - bot_pos[0]) / self.config.SCALE_X
        )
        screen_y = int(
            display_center_y - (pos[1] - bot_pos[1]) / self.config.SCALE_Y
        )  # Y инвертирован

        return screen_x, screen_y

    def update(
        self,
        current_time: float,
        bot_position: np.ndarray,
        bot_velocity: np.ndarray,
        bot_acceleration: np.ndarray,
        bot_orientation_quat: np.ndarray,
        bot_angular_velocity: np.ndarray,
        target_position: np.ndarray,
        sensor_data: Dict[str, Any],
        system_status: Dict[str, Any],
    ) -> None:
        """
        Обновляет и отрисовывает ASCII-визуализацию.

        Args:
            current_time (float): Текущее время симуляции.
            bot_position (np.ndarray): Текущая позиция бота.
            bot_velocity (np.ndarray): Текущая скорость бота.
            bot_acceleration (np.ndarray): Текущее ускорение бота.
            bot_orientation_quat (np.ndarray): Ориентация бота (кватернион).
            bot_angular_velocity (np.ndarray): Угловая скорость бота.
            target_position (np.ndarray): Текущая целевая позиция агента.
            sensor_data (Dict[str, Any]): Агрегированные данные всех датчиков.
            system_status (Dict[str, Any]): Агрегированные данные всех систем (питание, тепло и т.д.).
        """
        """
        if (time.time() - self.last_update_time) < self.config.UPDATE_INTERVAL:
            return
        """
        # Временно отключаем ограничение на частоту обновлений для тестирования

        self._clear_console()

        # Вычисляем координаты для отладочной информации
        bot_screen_x, bot_screen_y = self._world_to_screen(bot_position, bot_position)
        target_screen_x, target_screen_y = self._world_to_screen(
            target_position, bot_position
        )

        # Сбор информации для отображения
        display_lines = []
        display_lines.append(f"--- QIKI Simulation (T={current_time:.1f}s) ---")
        display_lines.append(f"Bot: [{bot_position[0]:.1f}, {bot_position[1]:.1f}, {bot_position[2]:.1f}] m")
        display_lines.append(f"Target: [{target_position[0]:.1f}, {target_position[1]:.1f}, {target_position[2]:.1f}] m")
        display_lines.append(f"Distance to target: {np.linalg.norm(target_position - bot_position):.1f} m")
        display_lines.append(
            f"Vel: [{bot_velocity[0]:.1f}, {bot_velocity[1]:.1f}, {bot_velocity[2]:.1f}] m/s (Mag: {np.linalg.norm(bot_velocity):.1f} m/s)"
        )
        if self.config.SHOW_ACCELERATION or self.display_mode == DisplayMode.DETAILED:
            display_lines.append(
                f"Acc: [{bot_acceleration[0]:.1f}, {bot_acceleration[1]:.1f}, {bot_acceleration[2]:.1f}] m/s^2"
            )
        display_lines.append(
            f"Target: [{target_position[0]:.1f}, {target_position[1]:.1f}, {target_position[2]:.1f}] m"
        )
        # Ориентация в виде кватерниона (x, y, z, w)
        display_lines.append(
            f"Quat: [{bot_orientation_quat[0]:.2f}, {bot_orientation_quat[1]:.2f}, "
            f"{bot_orientation_quat[2]:.2f}, {bot_orientation_quat[3]:.2f}]"
        )
        if self.config.SHOW_ANGULAR_VELOCITY or self.display_mode == DisplayMode.DETAILED:
            display_lines.append(
                f"AngVel: [{bot_angular_velocity[0]:.1f}, {bot_angular_velocity[1]:.1f}, {bot_angular_velocity[2]:.1f}] rad/s"
            )

        # Статус систем
        power_status = system_status.get("power", {})
        thermal_status = system_status.get("thermal", {})
        frame_status = system_status.get("frame", {})

        display_lines.append("--- System Status ---")
        display_lines.append(
            f"Power: {power_status.get('total_battery_percentage', 0):.1f}% "
            f"({power_status.get('status', 'N/A')})"
        )
        display_lines.append(
            f"Temp: {thermal_status.get('core_temperature_c', 0):.1f}°C "
            f"({thermal_status.get('status_message', 'N/A')})"
        )
        display_lines.append(
            f"Integrity: {frame_status.get('structural_integrity', 0):.1f}%"
        )
        display_lines.append(f"Mode: {self.display_mode.upper()}")


        # Дополнительная информация в зависимости от режима
        if self.display_mode == DisplayMode.DETAILED or self.display_mode == DisplayMode.DEBUG:
            display_lines.append("--- Sensor Readings (Sample) ---")
            if "power" in sensor_data:
                display_lines.append(f"  Solar Irradiance: {sensor_data['power'].get('solar_irradiance_w_m2', 0):.1f} W/m²")
            if "thermal" in sensor_data:
                display_lines.append(f"  Radiator Temp: {sensor_data['thermal'].get('radiator_temperature_c', 0):.1f}°C")
            if "frame" in sensor_data:
                display_lines.append(f"  Stress Level: {sensor_data['frame'].get('stress_level', 0):.2f}")

        if self.display_mode == DisplayMode.DEBUG:
            display_lines.append("--- Debug Info ---")
            display_lines.append("  Relative positions (screen coordinates):")
            display_lines.append(f"  Bot: ({bot_screen_x}, {bot_screen_y})")
            display_lines.append(f"  Target: ({target_screen_x}, {target_screen_y})")
            if hasattr(self, 'last_bot_pos'):
                dx = bot_position[0] - self.last_bot_pos[0]
                dy = bot_position[1] - self.last_bot_pos[1]
                dz = bot_position[2] - self.last_bot_pos[2]
                display_lines.append(f"  Movement since last update: [{dx:.2f}, {dy:.2f}, {dz:.2f}] m")
            self.last_bot_pos = bot_position.copy()


        # Объединяем и выводим информацию
        sys.stdout.write("\n".join(display_lines) + "\n")
        sys.stdout.flush()

        self.last_update_time = time.time()

    def close(self) -> None:
        """
        Метод для очистки ресурсов визуализатора (если необходимо).
        Для ASCII-визуализатора может быть пустым.
        """
        self.logger.info("AsciiVisualizer closed.")
        # Дополнительная очистка, если требуется