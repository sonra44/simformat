import numpy as np
from typing import Dict, Any

from logger import Logger
from physics import State
from ascii_visualizer import AsciiVisualizer as OriginalAsciiVisualizer

class AsciiVisualizerAdapter:
    """
    Адаптер для соединения асинхронного кода QIKI с исходным AsciiVisualizer.
    """
    
    def __init__(self, visualizer: OriginalAsciiVisualizer):
        """
        Инициализирует адаптер для ASCII-визуализатора.
        
        Args:
            visualizer: Экземпляр оригинального AsciiVisualizer
        """
        self.logger = Logger.get_logger("ascii_visualizer_adapter")
        self.visualizer = visualizer
        self.logger.info("AsciiVisualizerAdapter initialized.")
    
    def update(self, state: State, target_position: np.ndarray, sensor_data: Dict[str, Any], current_time: float = 0.0) -> None:
        """
        Адаптирует вызов update из main.py к интерфейсу оригинального AsciiVisualizer.
        
        Args:
            state: Текущее состояние физического объекта
            target_position: Целевая позиция
            sensor_data: Данные сенсоров
            current_time: Текущее время симуляции
        """
        # Соберем статус систем из sensor_data
        system_status = {
            "power": sensor_data.get("power", {}),
            "thermal": sensor_data.get("thermal", {}),
            "frame": sensor_data.get("frame", {})
        }
        
        # Вызываем оригинальный метод update с нужными параметрами
        self.visualizer.update(
            current_time=current_time,
            bot_position=state.position,
            bot_velocity=state.velocity,
            bot_acceleration=state.acceleration,
            bot_orientation_quat=state.orientation.as_quat(),
            bot_angular_velocity=state.angular_velocity,
            target_position=target_position,
            sensor_data=sensor_data,
            system_status=system_status
        )
    
    def close(self) -> None:
        """
        Закрывает визуализатор.
        """
        if self.visualizer:
            self.visualizer.close()
            self.logger.info("AsciiVisualizer closed via adapter.")