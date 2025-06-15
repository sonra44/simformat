import json
import numpy as np
import os
from datetime import datetime
from logger import Logger
from config import Config
from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from physics import PhysicsObject, State
    from agent import Agent

    # Sensors data is passed as a dict, so direct import of Sensors class isn't strictly needed for type hints here.


class Analyzer:
    """
    Класс Analyzer отвечает за сбор, хранение и анализ данных симуляции.
    Он записывает данные каждого шага и генерирует итоговый отчет.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Инициализирует анализатор, настраивает логгер и директорию для данных,
        а также создает файл сессии для сохранения результатов.

        Args:
            data_dir (str, optional): Директория для сохранения файлов сессии.
                                      По умолчанию "data".
        """
        self.logger: Logger = Logger.get_logger("analyzer")

        self.data_dir: str = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file: str = os.path.join(
            self.data_dir, f"session_{timestamp}.json"
        )

        self.session_data: Dict[str, Any] = {
            "start_time": timestamp,
            "steps": [],
            "statistics": {},  # Инициализируем поле для статистики
            "end_time": None,  # Инициализируем поле для времени окончания
        }

        self.logger.info(f"Analyzer initialized. Session file: {self.session_file}")

    def log_step(
        self,
        step: int,
        current_sim_time: float,  # Переименовано с 'time' для ясности и избежания конфликта с модулем time
        physics_obj: "PhysicsObject",
        sensors_data: Dict[str, Any],  # Переименовано с 'sensors' для ясности
        agent_obj: "Agent",  # Переименовано с 'agent' для ясности
    ) -> None:
        """
        Записывает данные текущего шага симуляции.

        Args:
            step (int): Номер текущего шага симуляции.
            current_sim_time (float): Текущее время симуляции.
            physics_obj (PhysicsObject): Объект PhysicsObject с текущим состоянием бота.
            sensors_data (Dict[str, Any]): Словарь с данными датчиков.
            agent_obj (Agent): Объект Agent с текущим статусом.
        """
        state: "State" = physics_obj.get_state()

        # Безопасное извлечение данных из sensors_data
        frame_info = sensors_data.get("frame", {})
        power_info = sensors_data.get("power", {})
        thermal_info = sensors_data.get("thermal", {})
        # navigation_info = sensors_data.get("navigation", {}) # Если нужно что-то специфичное отсюда

        step_data: Dict[str, Any] = {
            "step": step,
            "time": current_sim_time,
            "position": state.position.tolist(),
            "velocity": state.velocity.tolist(),
            "orientation": state.orientation.as_quat().tolist() if hasattr(state.orientation, 'as_quat') else state.orientation.tolist(),
            "angular_velocity": state.angular_velocity.tolist(),
            "integrity_percentage": frame_info.get("integrity_percentage", 0.0),
            "stress_level": frame_info.get("stress_level", 0.0),
            "main_battery_percentage": power_info.get("main_battery_percentage", 0.0),
            "reserve_battery_percentage": power_info.get(
                "reserve_battery_percentage", 0.0
            ),
            "total_battery_percentage": power_info.get("total_battery_percentage", 0.0),
            "solar_panel_output_w": power_info.get("solar_panel_output_w", 0.0),
            "current_draw_w": power_info.get("current_draw_w", 0.0),
            "core_temperature_c": thermal_info.get("core_temperature_c", 0.0),
            "agent_status": agent_obj.get_status(),
        }
        self.session_data["steps"].append(step_data)

    def calculate_session_statistics(self) -> Dict[str, Any]:
        """
        Рассчитывает и возвращает статистику за всю сессию симуляции.

        Returns:
            Dict[str, Any]: Словарь с рассчитанной статистикой сессии.
        """
        steps: List[Dict[str, Any]] = self.session_data.get("steps", [])
        if not steps:
            self.logger.warning(
                "No simulation steps recorded for statistics calculation."
            )
            return {}

        # Извлечение данных для расчетов
        positions = np.array([step.get("position", [0, 0, 0]) for step in steps])
        velocities = np.array([step.get("velocity", [0, 0, 0]) for step in steps])
        battery_percentages = [
            step.get("total_battery_percentage", 0.0) for step in steps
        ]
        solar_outputs = [step.get("solar_panel_output_w", 0.0) for step in steps]
        current_draws = [step.get("current_draw_w", 0.0) for step in steps]
        integrity_percentages = [
            step.get("integrity_percentage", 0.0) for step in steps
        ]
        stress_levels = [step.get("stress_level", 0.0) for step in steps]
        temperatures = [step.get("core_temperature_c", 0.0) for step in steps]
        times = [step.get("time", 0.0) for step in steps]

        total_distance_traveled_m = 0.0
        if len(positions) > 1:
            total_distance_traveled_m = float(
                np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            )

        max_velocity_mps = 0.0
        avg_velocity_mps = 0.0
        if velocities.size > 0:  # Проверка, что массив не пустой
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
            if velocity_magnitudes.size > 0:
                max_velocity_mps = float(np.max(velocity_magnitudes))
                avg_velocity_mps = float(np.mean(velocity_magnitudes))

        # Общее потребление энергии из батарей за симуляцию (Вт*ч)
        # Предполагаем, что current_draw_w - это мгновенное потребление на шаге
        # Интегрируем по времени
        total_power_demanded_wh = 0.0
        if len(times) > 1 and len(current_draws) == len(times):
            # Используем np.trapz для численного интегрирования (мощность * время)
            total_power_demanded_wh = float(np.trapz(y=current_draws, x=times) / 3600.0)

        stats: Dict[str, Any] = {
            "total_steps": len(steps),
            "simulation_duration_s": float(times[-1] - times[0]) if times else 0.0,
            "total_distance_traveled_m": total_distance_traveled_m,
            "max_velocity_mps": max_velocity_mps,
            "avg_velocity_mps": avg_velocity_mps,
            "total_power_demanded_wh": total_power_demanded_wh,
            "min_battery_percentage": (
                float(np.min(battery_percentages)) if battery_percentages else 0.0
            ),
            "avg_battery_percentage": (
                float(np.mean(battery_percentages)) if battery_percentages else 0.0
            ),
            "max_solar_output_w": (
                float(np.max(solar_outputs)) if solar_outputs else 0.0
            ),
            "avg_solar_output_w": (
                float(np.mean(solar_outputs)) if solar_outputs else 0.0
            ),
            "min_integrity_percentage": (
                float(np.min(integrity_percentages)) if integrity_percentages else 0.0
            ),
            "avg_integrity_percentage": (
                float(np.mean(integrity_percentages)) if integrity_percentages else 0.0
            ),
            "max_stress_level": float(np.max(stress_levels)) if stress_levels else 0.0,
            "avg_stress_level": float(np.mean(stress_levels)) if stress_levels else 0.0,
            "max_core_temperature_c": (
                float(np.max(temperatures)) if temperatures else 0.0
            ),
            "min_core_temperature_c": (
                float(np.min(temperatures)) if temperatures else 0.0
            ),
            "avg_core_temperature_c": (
                float(np.mean(temperatures)) if temperatures else 0.0
            ),
        }

        self.logger.info("Session statistics calculated.")
        return stats

    def finalize(self) -> None:
        """
        Завершает сессию анализатора, рассчитывает статистику и сохраняет
        все собранные данные и отчет в JSON файл.
        """
        self.session_data["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_data["statistics"] = self.calculate_session_statistics()

        try:
            with open(self.session_file, "w", encoding="utf-8") as f:
                json.dump(self.session_data, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Session data and report saved to {self.session_file}")
        except IOError as e:
            self.logger.error(f"Failed to save session file {self.session_file}: {e}")
        except TypeError as e:
            self.logger.error(f"Failed to serialize session data to JSON: {e}")
