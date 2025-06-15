"""
Адаптер для интеграции нового DisplayManager с существующей системой QIKI
"""

import numpy as np
import asyncio
from typing import Dict, Any

from display_manager import DisplayManager, DisplayFrame, DisplayConfig, InputCommand
from logger import Logger


class QikiDisplayAdapter:
    """
    Адаптер для интеграции DisplayManager с существующей архитектурой QIKI.
    """
    
    def __init__(self, qiki_simulation_instance=None):
        self.logger = Logger.get_logger("qiki_display_adapter")
        self.simulation = qiki_simulation_instance
        
        # Создаем и настраиваем DisplayManager
        config = DisplayConfig(
            width=120,
            height=35,
            update_rate=10.0,
            colors_enabled=True,
            show_fps=True
        )
        
        self.display_manager = DisplayManager(config)
        
        # Регистрируем колбэки для команд
        self._register_command_callbacks()
        
        self.logger.info("QikiDisplayAdapter initialized")
    
    def _register_command_callbacks(self):
        """Регистрирует колбэки для обработки команд."""
        
        async def pause_simulation():
            if self.simulation and hasattr(self.simulation, '_paused'):
                self.simulation._paused = True
                self.logger.info("Simulation paused")
        
        async def resume_simulation():
            if self.simulation and hasattr(self.simulation, '_paused'):
                self.simulation._paused = False
                self.logger.info("Simulation resumed")
        
        async def emergency_stop():
            if self.simulation:
                self.logger.warning("Emergency stop requested")
                if hasattr(self.simulation, 'agent') and self.simulation.agent:
                    self.simulation.agent.emergency_stop = True
        
        async def quit_simulation():
            if self.simulation and hasattr(self.simulation, '_running'):
                self.simulation._running = False
                self.logger.info("Quit requested")
        
        async def save_data():
            if self.simulation and hasattr(self.simulation, 'analyzer'):
                # Принудительно сохраняем данные
                self.simulation.analyzer.finalize()
                self.logger.info("Data saved")
        
        # Регистрируем колбэки
        self.display_manager.register_command_callback(InputCommand.PAUSE, pause_simulation)
        self.display_manager.register_command_callback(InputCommand.RESUME, resume_simulation)
        self.display_manager.register_command_callback(InputCommand.EMERGENCY_STOP, emergency_stop)
        self.display_manager.register_command_callback(InputCommand.QUIT, quit_simulation)
        self.display_manager.register_command_callback(InputCommand.SAVE_DATA, save_data)
    
    async def start(self):
        """Запускает DisplayManager."""
        await self.display_manager.start()
        self.logger.info("Display adapter started")
    
    def update(self, state, target_position, sensor_data, current_time):
        """
        Обновляет отображение. Совместим с интерфейсом ascii_visualizer_adapter.
        
        Args:
            state: Состояние физического объекта
            target_position: Целевая позиция
            sensor_data: Данные сенсоров
            current_time: Текущее время симуляции
        """
        # Собираем статус систем из sensor_data
        system_status = {
            "power": sensor_data.get("power", {}),
            "thermal": sensor_data.get("thermal", {}),
            "frame": sensor_data.get("frame", {})
        }
        
        # Получаем статус управления ботом
        control_status = None
        if hasattr(self.simulation, 'bot_control_integration'):
            control_status = self.simulation.bot_control_integration.get_control_status()
        
        # Создаем DisplayFrame
        frame = DisplayFrame(
            timestamp=current_time,
            bot_position=state.position,
            bot_velocity=state.velocity,
            bot_acceleration=state.acceleration,
            bot_orientation=state.orientation.as_quat(),
            target_position=target_position,
            system_status=system_status,
            sensor_data=sensor_data,
            simulation_state={},
            control_status=control_status
        )
        
        # Асинхронно обновляем дисплей
        asyncio.create_task(self.display_manager.update(frame))
    
    def close(self):
        """Закрывает DisplayManager."""
        self.display_manager.stop()
        self.logger.info("Display adapter closed")
    
    def set_display_mode(self, mode_name: str):
        """Устанавливает режим отображения по имени."""
        from display_manager import DisplayMode
        
        mode_mapping = {
            'minimal': DisplayMode.MINIMAL,
            'simplified': DisplayMode.STANDARD,  # Совместимость со старым API
            'standard': DisplayMode.STANDARD,
            'detailed': DisplayMode.DETAILED,
            'debug': DisplayMode.DEBUG,
            'telemetry': DisplayMode.TELEMETRY
        }
        
        if mode_name.lower() in mode_mapping:
            self.display_manager.set_mode(mode_mapping[mode_name.lower()])
        else:
            self.logger.warning(f"Unknown display mode: {mode_name}")
