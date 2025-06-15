# bot_interface_impl.py

from typing import Dict, Any, Optional, TYPE_CHECKING, List
import numpy as np
from qik_os import BotInterface
from logger import Logger
from config import config

if TYPE_CHECKING:
    from physics import PhysicsObject
    from sensors import Sensors
    from agent import Agent
    from hardware.frame_core import FrameCore
    from hardware.power_systems import PowerSystems
    from hardware.thermal_system import ThermalSystem
    from environment import Environment
    from main import QikiSimulation


class QikiBotInterface(BotInterface):
    """
    Конкретная реализация BotInterface для симуляции QIKI.
    Предоставляет QikOS доступ к компонентам симуляции.
    """
    def __init__(
        self,
        physics_obj: 'PhysicsObject',
        sensors_obj: 'Sensors',
        agent_obj: 'Agent',
        frame_core_obj: 'FrameCore',
        power_systems_obj: 'PowerSystems',
        thermal_system_obj: 'ThermalSystem',
        environment_obj: 'Environment',
        qiki_simulation_instance: Optional['QikiSimulation'] = None # Для доступа к current_sim_time
    ):
        self.logger = Logger.get_logger("bot_interface") # Используем наш Logger
        self.logger.info("BotInterface initialized.") # Добавил логирование инициализации

        self.physics_obj: 'PhysicsObject' = physics_obj
        self.sensors_obj: 'Sensors' = sensors_obj
        self.agent_obj: 'Agent' = agent_obj
        self.frame_core_obj: 'FrameCore' = frame_core_obj
        self.power_systems_obj: 'PowerSystems' = power_systems_obj
        self.thermal_system_obj: 'ThermalSystem' = thermal_system_obj
        self.environment_obj: 'Environment' = environment_obj

        # Это для доступа к текущему времени симуляции из QikiSimulation
        # Используется для демонстрации и логирования, не для прямой физики
        self._qiki_simulation_instance: Optional['QikiSimulation'] = qiki_simulation_instance
        self._current_sim_time: float = 0.0 # Локальное хранилище времени, если instance не передан

    async def get_bot_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус бота, агрегируя данные от всех подсистем.
        """
        full_sensor_data = self.sensors_obj.read_all()
        critical_status = self.sensors_obj.check_critical_status(full_sensor_data)

        # Добавим также статус агента и общие данные
        agent_status = self.agent_obj.get_status()

        return {
            "timestamp": self.get_current_simulation_time(),
            "physics": {
                "position": full_sensor_data.get("position"),
                "velocity": full_sensor_data.get("velocity"),
                "orientation": full_sensor_data.get("orientation"),
            },
            "frame": full_sensor_data.get("frame"),
            "power": full_sensor_data.get("power"),
            "thermal": full_sensor_data.get("thermal"),
            "agent": agent_status,
            "critical_status": critical_status,
            "environment": {
                "solar_irradiance": self.get_solar_irradiance(),
                # Можно добавить другие параметры среды, если они динамические
            }
        }

    async def set_thrust(self, x: float, y: float, z: float) -> str:
        """Устанавливает вектор тяги для бота."""
        thrust_vector = np.array([x, y, z])
        self.physics_obj.apply_force(thrust_vector)
        self.logger.info(f"Thrust set: {thrust_vector.tolist()}")
        return f"Thrust applied: {thrust_vector.tolist()}"

    async def set_torque(self, x: float, y: float, z: float) -> str:
        """Устанавливает вектор крутящего момента для бота."""
        torque_vector = np.array([x, y, z])
        self.physics_obj.apply_torque(torque_vector)
        self.logger.info(f"Torque set: {torque_vector.tolist()}")
        return f"Torque applied: {torque_vector.tolist()}"

    async def get_position(self) -> List[float]:
        """Возвращает текущую позицию бота."""
        return self.physics_obj.get_state().position.tolist()

    async def get_velocity(self) -> List[float]:
        """Возвращает текущую скорость бота."""
        return self.physics_obj.get_state().velocity.tolist()

    async def get_orientation(self) -> List[float]:
        """Возвращает текущую ориентацию бота (кватернион)."""
        return self.physics_obj.get_state().orientation.as_quat().tolist()

    async def set_target_position(self, x: float, y: float, z: float) -> str:
        """
        Устанавливает новую целевую позицию для автономного агента.
        """
        self.agent_obj.set_target(np.array([x, y, z]))
        return f"Target position set to: [{x}, {y}, {z}]"

    async def get_target_position(self) -> List[float]:
        """
        Возвращает текущую целевую позицию агента.
        """
        return self.agent_obj.target.tolist()

    async def enable_autonomous_mode(self) -> str:
        """Включает автономный режим агента."""
        self.agent_obj.set_autonomous_mode(True)
        return "Autonomous mode enabled."

    async def disable_autonomous_mode(self) -> str:
        """Выключает автономный режим агента."""
        self.agent_obj.set_autonomous_mode(False)
        return "Autonomous mode disabled."

    async def perform_emergency_stop(self) -> str:
        """Активирует аварийную остановку."""
        self.agent_obj.set_emergency_stop(True)
        # Возможно, также напрямую сбрасываем силы и моменты
        self.physics_obj.apply_force(np.zeros(3))
        self.physics_obj.apply_torque(np.zeros(3))
        return "Emergency stop activated. All thrusters/torques zeroed."

    async def clear_emergency_stop(self) -> str:
        """Деактивирует аварийную остановку."""
        self.agent_obj.set_emergency_stop(False)
        return "Emergency stop deactivated."

    async def get_current_simulation_time(self) -> float:
        """
        Возвращает текущее время симуляции.
        Предпочтительно брать из _qiki_simulation_instance, если он установлен.
        """
        if self._qiki_simulation_instance and hasattr(self._qiki_simulation_instance, 'current_simulation_time'):
            return self._qiki_simulation_instance.current_simulation_time
        # Если _qiki_simulation_instance не установлен, возвращаем локальное время,
        # которое должно обновляться извне (например, из QikiSimulation.run)
        return self._current_sim_time

    def update_simulation_time(self, current_sim_time: float):
        """Метод для QikiSimulation, чтобы обновлять время в интерфейсе."""
        self._current_sim_time = current_sim_time
        # self.logger.debug(f"Bot interface sim time updated to {current_sim_time}") # Отключено, чтобы не забивать логи

    def get_solar_irradiance(self) -> float:
        """Получает текущую солнечную иррадиацию из окружения."""
        # Убедимся, что physics_obj.get_state().position возвращает корректный np.ndarray
        current_position = self.physics_obj.get_state().position
        if not isinstance(current_position, np.ndarray) or current_position.shape != (3,):
            self.logger.warning(f"Invalid position from physics_obj for solar irradiance: {current_position}. Using default [0,0,0].")
            current_position = np.array([0.0, 0.0, 0.0]) # Fallback

        # Убедимся, что environment_obj имеет метод get_solar_irradiance
        if hasattr(self.environment_obj, 'get_solar_irradiance') and callable(getattr(self.environment_obj, 'get_solar_irradiance')):
            return self.environment_obj.get_solar_irradiance(current_position)
        else:
            self.logger.error("Environment object does not have 'get_solar_irradiance' method.")
            return config.DEFAULT_SOLAR_IRRADIANCE_W_M2 # Fallback to config value

    async def self_destruct(self) -> str:
        """
        Имитирует команду самоуничтожения бота.
        Эту команду нужно реализовать с большой осторожностью или использовать для симуляции отказа.
        """
        self.logger.critical("Initiating self-destruct sequence...")
        # В реальной симуляции здесь могут быть действия по остановке всех систем,
        # разрушению модели и т.д.
        # Для симуляции, возможно, просто завершаем работу приложения
        # (это должно быть обработано в main.py или через сигнал)
        # self._qiki_simulation_instance.shutdown() # Если есть прямой доступ и это безопасно
        return "Self-destruct sequence initiated. Farewell."

    def get_bot_position(self) -> np.ndarray:
        return self.physics_obj.state.position

    def get_bot_velocity(self) -> np.ndarray:
        return self.physics_obj.state.velocity

    def get_bot_acceleration(self) -> np.ndarray:
        return self.physics_obj.state.acceleration

    def get_bot_orientation_quat(self) -> np.ndarray:
        return self.physics_obj.state.orientation.as_quat()

    def get_bot_angular_velocity(self) -> np.ndarray:
        return self.physics_obj.state.angular_velocity

    def get_forward_vector(self) -> np.ndarray:
        return self.physics_obj.get_forward_vector()

    def get_gravity_vector(self) -> np.ndarray:
        return self.environment_obj.gravity

    def apply_thruster_force(self, force: np.ndarray) -> None:
        self.physics_obj.apply_force(force)

    def apply_torque(self, torque: np.ndarray) -> None:
        self.physics_obj.apply_torque(torque)

    def get_current_sim_time(self) -> float:
        if self._qiki_simulation_instance:
            return self._qiki_simulation_instance.current_sim_time
        return self._current_sim_time

    def get_aggregated_sensor_data(self) -> Dict[str, Any]:
        return self.sensors_obj.get_aggregated_data()

    def get_aggregated_system_status(self) -> Dict[str, Any]:
        return {
            'frame_core': self.frame_core_obj.get_status(),
            'power_systems': self.power_systems_obj.get_status(),
            'thermal_system': self.thermal_system_obj.get_status(),
        }

    def set_autonomous_mode(self, enabled: bool) -> None:
        self.agent_obj.autonomous = enabled
        self.logger.info(f"Autonomous mode {'enabled' if enabled else 'disabled'}")

    def set_new_target(self, target_position: np.ndarray) -> None:
        self.agent_obj.set_target(target_position)
        self.logger.info(f"New target set: {target_position}")

    def activate_emergency_stop(self) -> None:
        self.agent_obj.emergency_stop = True
        self.logger.warning("Emergency stop activated!")

    def deactivate_emergency_stop(self) -> None:
        self.agent_obj.emergency_stop = False
        self.logger.info("Emergency stop deactivated")