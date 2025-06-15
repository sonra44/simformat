#!/usr/bin/env python3
"""
QIKI Simulation - Чистая версия без избыточного вывода
=====================================================
"""

import time
import sys
import numpy as np
import asyncio
import os
from collections import deque
from typing import Dict, Any, Deque

from config import config
from physics import PhysicsObject
from agent import Agent
from sensors import Sensors
from environment import Environment
from logger import Logger
from analyzer import Analyzer
from hardware.frame_core import FrameCore
from hardware.power_systems import PowerSystems
from hardware.thermal_system import ThermalSystem
from qik_os import QikOS
from qiki_display_adapter import QikiDisplayAdapter
from qiki_bot_control_integration import QikiBotControlIntegration
from bot_interface_impl import QikiBotInterface


class QikiSimulationClean:
    """Чистая версия симуляции QIKI с минимальным выводом отладочной информации."""

    def __init__(self, enable_visualization=True):
        # Создаем директории
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        self.logger = Logger.get_logger("qiki_clean")
        self.logger.info("🚀 Starting QIKI Clean Simulation...")
        
        # Состояние симуляции
        self._running = False
        self._paused = False
        self.current_time = 0.0
        self.max_time = config.MAX_TIME
        
        # Инициализация компонентов
        self._init_components()
        
        # Дисплей и управление
        self.display_adapter = None
        self.bot_control_integration = None
        
        if enable_visualization:
            try:
                self.display_adapter = QikiDisplayAdapter(qiki_simulation_instance=self)
                self.logger.info("✅ Display adapter initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Display adapter failed: {e}")
        
        # Система управления ботом
        try:
            self.bot_control_integration = QikiBotControlIntegration(
                agent=self.agent,
                physics=self.physics,
                simulation=self
            )
            self.logger.info("✅ Bot control initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Bot control failed: {e}")

    def _init_components(self):
        """Инициализация основных компонентов."""
        # Физика
        from physics import State
        initial_state = State(
            position=np.array([0.0, 0.0, 5.0]),
            velocity=np.array([0.0, 0.0, 0.0])
        )
        self.physics = PhysicsObject(
            name="qiki_bot",
            mass=config.BOT_MASS_INITIAL,
            initial_state=initial_state
        )
        
        # Агент
        self.agent = Agent()
        
        # Аппаратура
        self.frame_core = FrameCore()
        self.power_systems = PowerSystems()
        self.thermal_system = ThermalSystem()
        
        # Сенсоры
        self.sensors = Sensors(
            physics_obj=self.physics,
            frame_core=self.frame_core,
            power_systems=self.power_systems,
            thermal_system=self.thermal_system
        )
        
        # Среда
        self.environment = Environment()
        
        # QikOS (создадим простой интерфейс)
        from qik_os import BotInterface
        
        class SimpleBotInterface(BotInterface):
            def __init__(self, physics, agent, sensors):
                self.physics = physics
                self.agent = agent
                self.sensors = sensors
                self.current_time = 0.0
            
            def get_current_sim_time(self) -> float:
                return self.current_time
            
            def get_bot_position(self) -> np.ndarray:
                return self.physics.get_state().position
            
            def get_bot_velocity(self) -> np.ndarray:
                return self.physics.get_state().velocity
            
            def get_bot_acceleration(self) -> np.ndarray:
                return self.physics.get_state().acceleration
            
            def get_bot_orientation_quat(self) -> np.ndarray:
                return self.physics.get_state().orientation.as_quat()
            
            def get_bot_angular_velocity(self) -> np.ndarray:
                return self.physics.get_state().angular_velocity
            
            def get_target_position(self) -> np.ndarray:
                return self.agent.target
            
            def get_aggregated_sensor_data(self) -> Dict[str, Any]:
                return {}
            
            def get_aggregated_system_status(self) -> Dict[str, Any]:
                return {}
            
            # Дополнительные методы
            def activate_emergency_stop(self):
                self.agent.emergency_stop = True
            
            def deactivate_emergency_stop(self):
                self.agent.emergency_stop = False
            
            def apply_thruster_force(self, force: np.ndarray):
                self.physics.apply_force(force)
            
            def apply_torque(self, torque: np.ndarray):
                self.physics.apply_torque(torque)
            
            def get_forward_vector(self) -> np.ndarray:
                return np.array([1.0, 0.0, 0.0])
            
            def get_gravity_vector(self) -> np.ndarray:
                return np.array([0.0, 0.0, -9.81])
            
            def get_solar_irradiance(self) -> float:
                return 1000.0
            
            def set_autonomous_mode(self, autonomous: bool):
                self.agent.autonomous = autonomous
            
            def set_new_target(self, target: np.ndarray):
                self.agent.target = target
        
        bot_interface = SimpleBotInterface(self.physics, self.agent, self.sensors)
        self.qik_os = QikOS(bot_interface)
        
        # Анализатор
        self.analyzer = Analyzer(data_dir="data")

    async def start(self):
        """Запуск симуляции."""
        if self._running:
            self.logger.warning("⚠️ Simulation already running")
            return

        self.logger.info("🎬 Starting simulation...")
        self._running = True

        try:
            # Инициализация
            if self.display_adapter:
                await self.display_adapter.start()
            
            if self.bot_control_integration:
                await self.bot_control_integration.start()
            
            # Запуск основного цикла
            await self._main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Simulation interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Simulation error: {e}")
        finally:
            await self.cleanup()

    async def _main_loop(self):
        """Основной цикл симуляции."""
        fixed_dt = 1.0 / config.SIMULATION_RATE
        accumulated_time = 0.0
        last_time = time.perf_counter()
        iteration = 0

        self.logger.info("🔄 Main loop started")

        while self.current_time < self.max_time and self._running:
            iteration += 1
            current_time = time.perf_counter()
            frame_time = current_time - last_time
            last_time = current_time
            
            accumulated_time += frame_time
            
            # Физические обновления
            update_count = 0
            while accumulated_time >= fixed_dt and update_count < 5:
                update_count += 1
                await self._update_simulation(fixed_dt)
                accumulated_time -= fixed_dt
            
            # Проверка критических состояний
            sensor_data = await self.sensors.read_all()
            critical_status = self.sensors.check_critical_status(sensor_data)
            
            if any(critical_status.values()):
                self.logger.critical("🚨 Critical status detected - emergency stop")
                await self.qik_os.shell.execute_command("emergency_stop", priority=0)
                break
            
            # Периодическое логирование (каждые 5 секунд)
            if iteration % (config.SIMULATION_RATE * 5) == 0:
                self.logger.info(f"⏱️ Time: {self.current_time:.1f}s | Updates: {update_count}")
            
            # Пауза для real-time режима
            if config.REAL_TIME_MODE:
                await asyncio.sleep(max(0, fixed_dt - frame_time))

        self.logger.info("🏁 Main loop completed")

    async def _update_simulation(self, dt):
        """Обновление симуляции."""
        # Обновление физики - применяем гравитацию как силу
        gravity_force = config.GRAVITY * self.physics.mass
        await self.physics.apply_force(gravity_force)
        
        # Обновление физики
        await self.physics.update(dt)
        
        # Получение данных сенсоров для агента
        sensor_data = await self.sensors.read_all()
        
        # Обновление агента
        await self.agent.update(sensor_data)
        
        # Обновление аппаратуры
        state = self.physics.get_state()
        self.frame_core.update(state, dt)
        self.power_systems.update(dt)
        self.thermal_system.update(self.power_systems.power_consumption, dt)
        
        # Обновление времени
        self.current_time += dt
        
        # Обновление дисплея (реже)
        if hasattr(self, '_last_display_update'):
            if self.current_time - self._last_display_update > 1.0:  # Раз в секунду
                await self._update_display()
                self._last_display_update = self.current_time
        else:
            self._last_display_update = self.current_time
            await self._update_display()

    async def _update_display(self):
        """Обновление дисплея."""
        if self.display_adapter:
            state = self.physics.get_state()
            sensor_data = await self.sensors.read_all()
            target = self.agent.target
            
            self.display_adapter.update(state, target, sensor_data, self.current_time)

    async def cleanup(self):
        """Очистка ресурсов."""
        self.logger.info("🧹 Cleaning up...")
        self._running = False
        
        if self.display_adapter:
            self.display_adapter.close()
        
        if self.bot_control_integration:
            self.bot_control_integration.stop()
        
        self.logger.info("✅ Cleanup completed")


async def main():
    """Главная функция."""
    print("🤖 QIKI Bot Simulation - Clean Version")
    print("=====================================")
    print("Press Ctrl+C to stop simulation")
    print()
    
    simulation = QikiSimulationClean(enable_visualization=True)
    await simulation.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Simulation stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
