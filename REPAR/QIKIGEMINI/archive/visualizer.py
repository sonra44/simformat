# visualizer.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import deque
from config import config
import time
from logger import Logger
from typing import Dict, Any, List, TYPE_CHECKING, Deque # Deque с большой буквы

if TYPE_CHECKING:
    from physics import State  # State импортируется для подсказок типов


class Visualizer:
    """
    Класс Visualizer отвечает за графическое отображение данных симуляции
    в реальном времени с использованием Matplotlib.
    """

    def __init__(self):
        """
        Инициализирует визуализатор, настраивает графики Matplotlib.
        """
        self.logger: Logger = Logger.get_logger("visualizer")
        plt.ion()  # Включаем интерактивный режим

        self.fig = plt.figure(figsize=(18, 12))
        self.fig.canvas.manager.set_window_title("QIKI Simulation Dashboard")

        # Определение осей для графиков
        self.ax_3d = self.fig.add_subplot(221, projection="3d")
        self.ax_power = self.fig.add_subplot(222)
        self.ax_integrity = self.fig.add_subplot(223)
        self.ax_temp = self.fig.add_subplot(224)

        self._setup_dashboard()  # Настройка внешнего вида дашборда

        # Данные телеметрии будут передаваться извне
        self.telemetry_data: Dict[str, Deque[Any]] = {}
        self.last_update_time: float = time.time() # Инициализация для корректного первого вызова

        self.logger.info("Visualizer initialized.")

    def _setup_dashboard(self) -> None:
        """
        Настраивает внешний вид и заголовки графиков на дашборде.
        """
        # 3D график позиции
        self.ax_3d.set_title("Bot Position (X, Y, Z)")
        self.ax_3d.set_xlabel("X (m)")
        self.ax_3d.set_ylabel("Y (m)")
        self.ax_3d.set_zlabel("Z (m)")
        self.ax_3d.set_xlim(config.BOUNDS["x"][0], config.BOUNDS["x"][1])
        self.ax_3d.set_ylim(config.BOUNDS["y"][0], config.BOUNDS["y"][1])
        self.ax_3d.set_zlim(config.BOUNDS["z"][0], config.BOUNDS["z"][1])
        self.ax_3d.grid(True)

        # График энергии
        self.ax_power.set_title("Power Systems (%)")
        self.ax_power.set_xlabel("Time (s)")
        self.ax_power.set_ylabel("Battery Charge (%)")
        self.ax_power.set_ylim(0, 105)
        self.ax_power.grid(True)

        # График структурной целостности
        self.ax_integrity.set_title("Structural Integrity (%)")
        self.ax_integrity.set_xlabel("Time (s)")
        self.ax_integrity.set_ylabel("Integrity (%)")
        self.ax_integrity.set_ylim(-5, 105)
        self.ax_integrity.grid(True)

        # График температуры
        self.ax_temp.set_title("Core Temperature (°C)")
        self.ax_temp.set_xlabel("Time (s)")
        self.ax_temp.set_ylabel("Temperature (°C)")
        self.ax_temp.grid(True)


    def _update_dashboard(self, target_position: np.ndarray) -> None:
        """
        Обновляет данные на всех графиках дашборда.
        """
        # Очищаем все графики
        self.ax_3d.clear()
        self.ax_power.clear()
        self.ax_integrity.clear()
        self.ax_temp.clear()

        self._setup_dashboard() # Перенастраиваем заголовки и метки после очистки

        time_data = list(self.telemetry_data.get("time", deque()))
        if not time_data:
            self.logger.debug("No time data available for visualization.")
            return # Нет данных для отображения

        # 3D график позиции
        pos_x = list(self.telemetry_data.get("position_x", deque()))
        pos_y = list(self.telemetry_data.get("position_y", deque()))
        pos_z = list(self.telemetry_data.get("position_z", deque()))
        
        # Проверяем, что у нас есть данные и их размерности совпадают
        if pos_x and pos_y and pos_z and len(pos_x) == len(pos_y) == len(pos_z):
            self.ax_3d.plot(pos_x, pos_y, pos_z, label="Bot Path", color='blue')
            self.ax_3d.scatter(pos_x[-1], pos_y[-1], pos_z[-1], color='red', marker='o', s=50, label="Current Position")
            self.ax_3d.scatter(target_position[0], target_position[1], target_position[2], color='green', marker='x', s=100, label="Target")
            self.ax_3d.legend()

        # График энергии
        main_batt = list(self.telemetry_data.get("main_battery_percentage", deque()))
        total_batt = list(self.telemetry_data.get("battery_percentage", deque()))  # Используем battery_percentage вместо total_battery_percentage
        solar_output = list(self.telemetry_data.get("solar_irradiance", deque()))  # Используем solar_irradiance вместо solar_panel_output_w
        power_draw = list(self.telemetry_data.get("total_power_draw_w", deque()) or deque())

        if time_data and main_batt and len(main_batt) == len(time_data):
            self.ax_power.plot(time_data, main_batt, label="Main Battery (%)", color='green')
        if time_data and total_batt and len(total_batt) == len(time_data):
            self.ax_power.plot(time_data, total_batt, label="Total Battery (%)", color='blue', linestyle='--')
            
        if time_data and (solar_output or power_draw):
            ax2_power = self.ax_power.twinx()
            ax2_power.set_ylabel("Power (W)", color='purple')
            
            if solar_output and len(solar_output) == len(time_data):
                ax2_power.plot(time_data, solar_output, label="Solar Irradiance (W/m²)", color='orange', linestyle=':')
            
            if power_draw and len(power_draw) == len(time_data):
                ax2_power.plot(time_data, power_draw, label="Total Draw (W)", color='red', linestyle='-.')
                
            ax2_power.tick_params(axis='y', labelcolor='purple')
            lines, labels = self.ax_power.get_legend_handles_labels()
            lines2, labels2 = ax2_power.get_legend_handles_labels()
            self.ax_power.legend(lines + lines2, labels + labels2, loc="upper left")

        # График структурной целостности
        integrity_data = list(self.telemetry_data.get("structural_integrity", deque()))
        stress_data = list(self.telemetry_data.get("stress_level", deque()))
        
        if time_data and integrity_data and len(integrity_data) == len(time_data):
            self.ax_integrity.plot(time_data, integrity_data, label="Structural Integrity (%)", color='teal')
            
            if stress_data and len(stress_data) == len(time_data):
                ax2_integrity = self.ax_integrity.twinx()
                ax2_integrity.set_ylabel("Stress Level", color='brown')
                ax2_integrity.plot(time_data, stress_data, label="Stress Level", color='brown', linestyle=':')
                ax2_integrity.set_ylim(-0.1, 1.1)
                ax2_integrity.tick_params(axis='y', labelcolor='brown')
                lines, labels = self.ax_integrity.get_legend_handles_labels()
                lines2, labels2 = ax2_integrity.get_legend_handles_labels()
                self.ax_integrity.legend(lines + lines2, labels + labels2, loc="upper right")

        # График температуры
        core_temp = list(self.telemetry_data.get("core_temperature_c", deque()))
        radiator_temp = list(self.telemetry_data.get("radiator_temperature_c", deque()))
        
        if time_data and core_temp and len(core_temp) == len(time_data):
            self.ax_temp.plot(time_data, core_temp, label="Core Temp (°C)", color='darkred')
            
            if radiator_temp and len(radiator_temp) == len(time_data):
                self.ax_temp.plot(time_data, radiator_temp, label="Radiator Temp (°C)", color='darkblue')
                
            # Добавим линии критических порогов
            self.ax_temp.axhline(y=config.OVERHEAT_THRESHOLD_C, color='red', linestyle='--', label='Overheat Threshold')
            self.ax_temp.axhline(y=config.FREEZE_THRESHOLD_C, color='cyan', linestyle='--', label='Freeze Threshold')
            
            # Вычисляем лимиты оси Y с защитой от пустых массивов
            all_temps = []
            if core_temp:
                all_temps.extend(core_temp)
            if radiator_temp:
                all_temps.extend(radiator_temp)
            if all_temps:
                all_temps.extend([config.FREEZE_THRESHOLD_C, config.OVERHEAT_THRESHOLD_C])
                self.ax_temp.set_ylim(min(all_temps) - 5, max(all_temps) + 5)
            
            # Установка динамических пределов X для всех графиков
            # Все графики имеют общую ось времени, поэтому xlim должен быть одинаковым
            if time_data:
                start_time = time_data[0]
                end_time = time_data[-1]
                self.ax_3d.set_xlim(config.BOUNDS["x"][0], config.BOUNDS["x"][1])  # Остаются статичными для 3D
                self.ax_power.set_xlim(start_time, end_time + 1)
                self.ax_integrity.set_xlim(start_time, end_time + 1)
                self.ax_temp.set_xlim(start_time, end_time + 1)
                self.ax_temp.legend(loc="upper right")  # Обновляем легенду, чтобы она не пропадала


        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)  # Добавляем небольшую паузу для обработки событий Matplotlib
        except Exception as e:
            self.logger.warning(f"Error during plot drawing: {e}")

    def update(
        self,
        telemetry_data: Dict[str, Deque[Any]],
        target_position: np.ndarray
    ) -> None:
        """
        Основной метод обновления визуализатора.
        Вызывается из главного цикла симуляции.

        Args:
            telemetry_data (Dict[str, Deque]): Словарь с очередями данных телеметрии от QikOS.
            target_position (np.ndarray): Текущая цель агента.
        """
        # ИЗМЕНЕНИЕ: Исправлено имя константы UPDATE_RATE на UPDATE_RATE_VISUALIZER_FPS
        if (time.time() - self.last_update_time) < (1.0 / config.UPDATE_RATE_VISUALIZER_FPS):
            return

        try:
            self.telemetry_data = telemetry_data
            self._update_dashboard(target_position)
        except Exception as e:
            self.logger.error(f"Error in Visualizer update: {e}", exc_info=True)

        self.last_update_time = time.time()

    def close(self) -> None:
        """
        Закрывает все окна Matplotlib.
        """
        self.logger.info("Closing Visualizer.")
        plt.close(self.fig) # Закрываем конкретную фигуру
        plt.close('all') # Закрываем все оставшиеся фигуры Matplotlib
        self.logger.info("Visualizer closed.")