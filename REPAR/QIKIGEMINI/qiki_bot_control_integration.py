"""
Интеграция системы управления ботом с основной симуляцией QIKI
"""

import asyncio
import numpy as np
from typing import Optional

from bot_control import BotController, ControlMode, ControlState
from logger import Logger


class QikiBotControlIntegration:
    """
    Интеграция системы управления ботом с симуляцией QIKI.
    Обеспечивает связь между пользовательскими командами и агентом.
    """
    
    def __init__(self, agent=None, physics=None, simulation=None):
        self.logger = Logger.get_logger("qiki_bot_control")
        self.agent = agent
        self.physics = physics
        self.simulation = simulation
        
        # Создаем контроллер бота
        self.bot_controller = BotController(agent=agent, physics=physics)
        
        # Флаги состояния
        self.is_active = False
        self.manual_override = False
        
        self.logger.info("QikiBotControlIntegration initialized")
    
    async def start(self):
        """Запускает систему управления."""
        try:
            await self.bot_controller.start()
            self.is_active = True
            
            # Запускаем задачу интеграции
            asyncio.create_task(self._integration_loop())
            
            self.logger.info("Bot control system started")
            self._show_startup_info()
            
        except Exception as e:
            self.logger.error(f"Failed to start bot control: {e}")
    
    def _show_startup_info(self):
        """Показывает информацию о запуске системы управления."""
        print("\\n" + "="*60)
        print("🤖 QIKI BOT CONTROL SYSTEM ACTIVE")
        print("="*60)
        print("Available control methods:")
        
        # Показываем доступные интерфейсы
        for interface in self.bot_controller.active_interfaces:
            interface_name = interface.__class__.__name__
            if "Keyboard" in interface_name:
                print("  ⌨️  Keyboard Control - Use WASD, IJKL for movement")
            elif "Text" in interface_name:
                print("  💬 Text Commands - Type commands like 'forward', 'goto 10 5 15'")
            elif "Touch" in interface_name:
                print("  📱 Touch Control - Virtual joystick available")
        
        print("\\nCurrent mode: AUTONOMOUS (use 'manual' command to switch)")
        print("Emergency stop: Press X or type 'emergency'")
        print("="*60 + "\\n")
    
    async def _integration_loop(self):
        """
        Основной цикл интеграции - связывает команды управления с агентом.
        """
        while self.is_active:
            try:
                # Получаем состояние управления
                control_state = self.bot_controller.get_control_state()
                
                # Обрабатываем экстренную остановку
                if control_state.emergency_stop and self.agent:
                    self.agent.emergency_stop = True
                    if self.simulation:
                        self.simulation._running = False
                
                # Обрабатываем смену режимов
                if control_state.mode == ControlMode.MANUAL:
                    if self.agent and self.agent.autonomous:
                        self.agent.autonomous = False
                        self.manual_override = True
                        self.logger.info("Switched to MANUAL control")
                
                elif control_state.mode == ControlMode.AUTONOMOUS:
                    if self.agent and not self.agent.autonomous:
                        self.agent.autonomous = True
                        self.manual_override = False
                        self.logger.info("Switched to AUTONOMOUS control")
                
                # В ручном режиме применяем команды напрямую
                if self.manual_override and self.physics:
                    await self._apply_manual_control(control_state)
                
                # Небольшая пауза
                await asyncio.sleep(0.02)  # 50 Hz
                
            except Exception as e:
                self.logger.error(f"Integration loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _apply_manual_control(self, control_state: ControlState):
        """
        Применяет ручное управление к физическому объекту.
        """
        try:
            # Получаем текущее состояние
            physics_state = self.physics.get_state()
            
            # Применяем тягу (силы)
            if np.any(control_state.thrust_vector):
                # Преобразуем команды скорости в силы
                force = control_state.thrust_vector * self.physics.mass * 2.0  # Усиление
                
                # Применяем силу в мировых координатах
                self.physics.apply_force(force)
            
            # Применяем вращение (моменты)
            if np.any(control_state.torque_vector):
                # Преобразуем команды угловой скорости в моменты
                torque = control_state.torque_vector * 10.0  # Усиление
                
                # Применяем момент
                self.physics.apply_torque(torque)
            
            # Обновляем целевую позицию агента, если задана
            if control_state.target_position is not None and self.agent:
                self.agent.target = control_state.target_position
            
        except Exception as e:
            self.logger.error(f"Error applying manual control: {e}")
    
    def get_control_status(self) -> dict:
        """Возвращает статус системы управления для отображения."""
        if not self.is_active:
            return {"status": "inactive"}
        
        control_state = self.bot_controller.get_control_state()
        
        return {
            "status": "active",
            "mode": control_state.mode.name,
            "manual_override": self.manual_override,
            "emergency_stop": control_state.emergency_stop,
            "last_command": control_state.last_command_time,
            "thrust_vector": control_state.thrust_vector.tolist(),
            "torque_vector": control_state.torque_vector.tolist(),
            "active_interfaces": len(self.bot_controller.active_interfaces),
            "command_history_size": len(self.bot_controller.command_history)
        }
    
    def stop(self):
        """Останавливает систему управления."""
        self.is_active = False
        self.bot_controller.stop()
        self.logger.info("Bot control integration stopped")
