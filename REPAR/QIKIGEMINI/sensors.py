import numpy as np
import asyncio
from collections import deque
from config import config
from logger import Logger
from typing import Dict, Any, Tuple, Optional, TYPE_CHECKING, Deque

if TYPE_CHECKING:
    from physics import PhysicsObject
    from hardware.frame_core import FrameCore
    from hardware.power_systems import PowerSystems
    from hardware.thermal_system import ThermalSystem


class Sensors:
    """
    Асинхронный класс Sensors собирает и агрегирует показания датчиков бота
    от различных аппаратных модулей и физического состояния.
    Поддерживает периодическое обновление и кэширование данных.
    """

    def __init__(
        self,
        physics_obj: 'PhysicsObject',
        frame_core: 'FrameCore',
        power_systems: 'PowerSystems',
        thermal_system: 'ThermalSystem',
    ):
        """
        Инициализирует асинхронную систему сенсоров.

        Args:
            physics_obj: Экземпляр физического объекта бота
            frame_core: Экземпляр системы каркаса
            power_systems: Экземпляр энергетической системы
            thermal_system: Экземпляр термической системы
        """
        self.logger: Logger = Logger.get_logger("sensors")
        self.physics_obj = physics_obj
        self.frame_core = frame_core
        self.power_systems = power_systems
        self.thermal_system = thermal_system

        # События и флаги
        self.update_event = asyncio.Event()
        self.running = True
        self.is_updating = False

        # Кэш данных сенсоров с временными метками
        self._sensor_cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_lifetime = 0.1  # Время жизни кэша в секундах

        # История показаний для анализа трендов
        self._history: Dict[str, Deque[Tuple[float, Any]]] = {
            "navigation": deque(maxlen=100),
            "power": deque(maxlen=100),
            "thermal": deque(maxlen=100),
            "frame": deque(maxlen=100)
        }

        # Последние примененные силы
        self.last_forces_applied: Tuple[np.ndarray, np.ndarray] = (
            np.zeros(3),
            np.zeros(3)
        )

        self.logger.info("Async Sensors initialized with all subsystems")

    async def update(self):
        """
        Асинхронно обновляет данные сенсоров. Метод можно вызывать периодически
        для обновления данных в фоновом режиме.
        """
        while self.running:
            await self.update_event.wait()  # Ждет, пока событие не будет установлено
            self.is_updating = True

            # Обновление данных сенсоров
            self._sensor_cache["navigation"] = (self._current_time(), self._read_navigation())
            self._sensor_cache["power"] = (self._current_time(), self._read_power())
            self._sensor_cache["thermal"] = (self._current_time(), self._read_thermal())
            self._sensor_cache["frame"] = (self._current_time(), self._read_frame_integrity())

            # Очистка устаревшего кэша
            self._clear_cache()

            self.is_updating = False
            self.update_event.clear()  # Сбрасывает событие

    def _current_time(self) -> float:
        """
        Возвращает текущее время в секундах с начала эпохи.
        Используется для временных меток в кэше сенсоров.
        """
        return asyncio.get_event_loop().time()

    def _clear_cache(self):
        """
        Очищает устаревшие записи в кэше сенсоров на основе времени жизни кэша.
        """
        current_time = self._current_time()
        for key in list(self._sensor_cache.keys()):
            if current_time - self._sensor_cache[key][0] > self._cache_lifetime:
                del self._sensor_cache[key]



        sensor_data: Dict[str, Any] = {
            "navigation": self._read_navigation(),
            "power": self._read_power(),
            "thermal": self._read_thermal(),
            "frame": self._read_frame_integrity(),
        }
        
        # Дополнительная агрегация или вычисления могут быть добавлены здесь, если необходимо.
        # Например, общий статус системы на основе всех датчиков.

        return sensor_data

    async def read_all(self, forces_applied: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Асинхронно собирает все доступные показания датчиков.
        Использует кэширование для оптимизации производительности.

        Args:
            forces_applied: Последние примененные силы (опционально)

        Returns:
            Dict с агрегированными данными всех сенсоров
        """
        try:
            if forces_applied:
                self.last_forces_applied = forces_applied

            # Собираем данные от всех подсистем параллельно
            tasks = [
                self._read_navigation(),
                self._read_power(),
                self._read_thermal(),
                self._read_frame_integrity()
            ]
            
            navigation, power, thermal, frame = await asyncio.gather(*tasks)
            
            sensor_data = {
                "navigation": navigation,
                "power": power,
                "thermal": thermal,
                "frame": frame,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Обновляем историю
            for key, value in sensor_data.items():
                if key in self._history:
                    self._history[key].append((sensor_data["timestamp"], value))
            
            # Оповещаем об обновлении
            self.update_event.set()
            
            return sensor_data
            
        except Exception as e:
            self.logger.error(f"Error reading sensors: {str(e)}")
            raise

    async def _read_navigation(self) -> Dict[str, Any]:
        """
        Асинхронно считывает навигационные данные.

        Returns:
            Dict с навигационными данными
        """
        # Проверяем кэш
        cache_key = "navigation"
        current_time = asyncio.get_event_loop().time()
        
        if cache_key in self._sensor_cache:
            timestamp, data = self._sensor_cache[cache_key]
            if current_time - timestamp < self._cache_lifetime:
                return data
        
        # Получаем свежие данные без блокирующего ожидания обновления физики
        # await self.physics_obj.wait_for_update()  # Убираем блокирующее ожидание
        # Вместо этого сразу получаем текущее состояние
        state = self.physics_obj.get_state()
        
        nav_data = {
            "position": state.position.copy(),
            "velocity": state.velocity.copy(),
            "acceleration": state.acceleration.copy(),
            "angular_velocity": state.angular_velocity.copy(),
            "altitude": float(state.position[2]),
            "speed": float(np.linalg.norm(state.velocity)),
            "angular_speed": float(np.linalg.norm(state.angular_velocity)),
        }
        
        # Кэшируем результат
        self._sensor_cache[cache_key] = (current_time, nav_data)
        return nav_data

    async def _read_power(self) -> Dict[str, Any]:
        """
        Асинхронно считывает данные энергосистемы.

        Returns:
            Dict с данными энергосистемы
        """
        cache_key = "power"
        current_time = asyncio.get_event_loop().time()
        
        if cache_key in self._sensor_cache:
            timestamp, data = self._sensor_cache[cache_key]
            if current_time - timestamp < self._cache_lifetime:
                return data
        
        power_data = {
            "total_battery_percentage": self.power_systems.get_total_battery_percentage(),
            "power_consumption": self.power_systems.get_power_consumption(),
            "solar_power": self.power_systems.get_solar_power(),
            "charging_state": self.power_systems.get_charging_state()
        }
        
        self._sensor_cache[cache_key] = (current_time, power_data)
        return power_data

    async def _read_thermal(self) -> Dict[str, Any]:
        """
        Асинхронно считывает данные термической системы.

        Returns:
            Dict с термическими данными
        """
        cache_key = "thermal"
        current_time = asyncio.get_event_loop().time()
        
        if cache_key in self._sensor_cache:
            timestamp, data = self._sensor_cache[cache_key]
            if current_time - timestamp < self._cache_lifetime:
                return data
                
        # Получаем данные через метод get_status()
        thermal_status = self.thermal_system.get_status()
        thermal_data = {
            "core_temperature_c": thermal_status["core_temperature_c"],
            "radiator_temperature_c": thermal_status["radiator_temperature_c"],
            "thermal_status": thermal_status["status_message"],
            "power_consumption_w": self.thermal_system.get_power_consumption()
        }
        
        self._sensor_cache[cache_key] = (current_time, thermal_data)
        return thermal_data

    async def _read_frame_integrity(self) -> Dict[str, Any]:
        """
        Асинхронно считывает данные о состоянии каркаса.

        Returns:
            Dict с данными о целостности каркаса
        """
        cache_key = "frame"
        current_time = asyncio.get_event_loop().time()
        
        if cache_key in self._sensor_cache:
            timestamp, data = self._sensor_cache[cache_key]
            if current_time - timestamp < self._cache_lifetime:
                return data
                
        # Получаем данные через метод get_status()
        frame_status = self.frame_core.get_status()
        sensors_data = frame_status["sensors_data"]
        
        frame_data = {
            "integrity_percentage": frame_status["structural_integrity"],
            "stress_level": sensors_data["stress_level"],
            "vibration_level": sensors_data["vibration_level"],
            "deformation": 0.0  # Пока не имеем данных о деформации, используем заглушку
        }
        
        self._sensor_cache[cache_key] = (current_time, frame_data)
        return frame_data

    def check_critical_status(self, sensor_data: Dict[str, Any]) -> Dict[str, bool]:
        """
        Проверяет критические состояния на основе данных сенсоров.
        
        Args:
            sensor_data: Данные сенсоров, полученные от read_all()
            
        Returns:
            Словарь критических состояний {состояние: True/False}
        """
        result = {
            "low_battery": False,
            "overheating": False,
            "critical_integrity": False,
            "out_of_bounds": False
        }
        
        # Проверка уровня заряда батареи
        power_data = sensor_data.get("power", {})
        battery_percentage = power_data.get("total_battery_percentage", 100.0)
        
        if battery_percentage <= config.LOW_BATTERY_THRESHOLD:
            result["low_battery"] = True
            self.logger.warning(f"Low battery: {battery_percentage:.1f}%")
            
        # Проверка температуры
        thermal_data = sensor_data.get("thermal", {})
        core_temp = thermal_data.get("core_temperature_c", 25.0)
        
        if core_temp >= config.OVERHEAT_THRESHOLD_C:
            result["overheating"] = True
            self.logger.warning(f"Overheating: {core_temp:.1f}°C")
            
        # Проверка структурной целостности
        frame_data = sensor_data.get("frame", {})
        integrity = frame_data.get("structural_integrity", 100.0)
        
        if integrity <= config.CRITICAL_INTEGRITY_THRESHOLD:
            result["critical_integrity"] = True
            self.logger.warning(f"Critical integrity: {integrity:.1f}%")
            
        # Проверка позиции (за пределами допустимой области)
        nav_data = sensor_data.get("navigation", {})
        position = nav_data.get("position", [0, 0, 0])
        
        # Исправлено для использования правильного формата ключей в config.BOUNDS
        if (
            position[0] < config.BOUNDS["x"][0] or position[0] > config.BOUNDS["x"][1] or
            position[1] < config.BOUNDS["y"][0] or position[1] > config.BOUNDS["y"][1] or
            position[2] < config.BOUNDS["z"][0] or position[2] > config.BOUNDS["z"][1]
        ):
            result["out_of_bounds"] = True
            self.logger.warning(f"Out of bounds: position {position}")
            
        return result

    def get_all_sensor_data(self) -> Dict[str, Any]:
        """
        Синхронно возвращает все данные сенсоров из кэша.
        Используется для совместимости со старым кодом.
        
        Returns:
            Dict[str, Any]: Словарь с данными всех сенсоров
        """
        result = {}
        
        # Берем данные из кэша
        for key, (_, data) in self._sensor_cache.items():
            result[key] = data
            
        # Если какие-то данные отсутствуют, возвращаем пустые словари
        if "navigation" not in result:
            result["navigation"] = {}
        if "power" not in result:
            result["power"] = {}
        if "thermal" not in result:
            result["thermal"] = {}
        if "frame" not in result:
            result["frame"] = {}
            
        return result

    async def start_updates(self, update_interval: float = 0.1) -> None:
        """
        Запускает асинхронное обновление датчиков с заданным интервалом.

        Args:
            update_interval: Интервал обновления в секундах
        """
        self.is_updating = True
        while self.running and self.is_updating:
            try:
                await self.read_all()
                await asyncio.sleep(update_interval)
            except Exception as e:
                self.logger.error(f"Error in sensor update loop: {str(e)}")
                await asyncio.sleep(1.0)  # Увеличенная задержка при ошибке

    async def stop_updates(self) -> None:
        """Останавливает асинхронное обновление датчиков"""
        self.is_updating = False
        self.logger.info("Sensor updates stopped")

    def get_history(self, sensor_type: str, duration: float = None) -> list:
        """
        Возвращает историю показаний определенного типа датчиков.

        Args:
            sensor_type: Тип датчика ("navigation", "power", "thermal", "frame")
            duration: Длительность истории в секундах (None для всей истории)

        Returns:
            list: История показаний датчика
        """
        if sensor_type not in self._history:
            return []

        history = list(self._history[sensor_type])
        if not duration:
            return history

        current_time = asyncio.get_event_loop().time()
        return [
            (t, v) for t, v in history
            if current_time - t <= duration
        ]

    async def reset(self) -> None:
        """
        Сбрасывает состояние всех датчиков и очищает историю.
        """
        # Останавливаем обновления
        was_updating = self.is_updating
        if was_updating:
            await self.stop_updates()

        # Очищаем кэш и историю
        self._sensor_cache.clear()
        for queue in self._history.values():
            queue.clear()

        # Сбрасываем силы
        self.last_forces_applied = (np.zeros(3), np.zeros(3))

        # Восстанавливаем обновления, если были активны
        if was_updating:
            await self.start_updates()

        self.logger.info("Sensors reset completed")