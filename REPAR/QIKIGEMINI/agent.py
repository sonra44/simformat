import numpy as np
import time
import asyncio
from collections import deque
from logger import Logger
from config import config
from typing import Tuple, Dict, Any, Deque


class Agent:
    """
    Асинхронный агент принятия решений для бота QIKI.
    Анализирует данные сенсоров и управляет поведением бота в реальном времени.
    """

    def __init__(self):
        """
        Инициализирует асинхронного агента с поддержкой событий и очередей.
        """
        self.logger: Logger = Logger.get_logger("agent")
        
        # Целевая позиция и состояние
        self.target: np.ndarray = np.array([10.0, 10.0, 10.0])
        self.emergency_stop: bool = False
        self.autonomous: bool = True
        
        # Временные параметры
        self.last_decision_time: float = 0.0
        self.decision_interval: float = 0.5
        
        # События и очереди
        self.decision_event = asyncio.Event()
        self.command_queue: asyncio.Queue = asyncio.Queue()
        self.running: bool = True
        
        # История команд и состояний
        self.command_history: Deque[Tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=100)
        self.state_history: Deque[Dict[str, Any]] = deque(maxlen=100)
        
        # Параметры управления
        self.max_thrust = config.MAX_THRUST_N
        self.max_torque = config.MAX_TORQUE_NM
        self.pid_gains = {
            "position": {"P": 1.0, "I": 0.1, "D": 0.5},
            "attitude": {"P": 2.0, "I": 0.2, "D": 1.0}
        }
        
        # Интеграторы для PID
        self.position_error_integral = np.zeros(3)
        self.attitude_error_integral = np.zeros(3)
        
        self.logger.info("Async Agent initialized with event system and command queue")

    async def _engage_emergency_stop(self) -> None:
        """
        Асинхронно активирует аварийную остановку.
        """
        self.emergency_stop = True
        self.autonomous = False
        # Отправляем команду остановки
        await self.command_queue.put((np.zeros(3), np.zeros(3)))
        self.logger.warning("Emergency stop engaged")
        
    async def start(self) -> None:
        """
        Запускает асинхронную работу агента.
        """
        self.running = True
        self.emergency_stop = False
        self.logger.info("Agent started")
        
    async def stop(self) -> None:
        """
        Останавливает работу агента.
        """
        self.running = False
        await self._engage_emergency_stop()
        # Очищаем очередь команд
        while not self.command_queue.empty():
            await self.command_queue.get()
        self.logger.info("Agent stopped")
        
    async def set_autonomous_mode(self, mode: bool) -> None:
        """
        Асинхронно переключает автономный режим.
        
        Args:
            mode: True для включения автономного режима
        """
        self.autonomous = mode
        if not mode:
            # При выключении автономного режима отправляем команду остановки
            await self.command_queue.put((np.zeros(3), np.zeros(3)))
        self.logger.info(f"Autonomous mode: {mode}")
        
    async def set_target(self, new_target: np.ndarray) -> None:
        """
        Асинхронно устанавливает новую цель.
        
        Args:
            new_target: Новая целевая позиция
        """
        if not isinstance(new_target, np.ndarray) or new_target.shape != (3,):
            self.logger.error(f"Invalid target format: {new_target}")
            return
            
        self.target = new_target.copy()
        # Сбрасываем интеграторы PID при новой цели
        self.position_error_integral[:] = 0
        self.attitude_error_integral[:] = 0
        self.logger.info(f"New target set: {self.target}")
        
    async def reset(self) -> None:
        """
        Асинхронно сбрасывает состояние агента.
        """
        # Сбрасываем флаги и параметры
        self.emergency_stop = False
        self.autonomous = True
        self.target = np.array([10.0, 10.0, 10.0])
        self.last_decision_time = 0.0
        
        # Сбрасываем интеграторы
        self.position_error_integral[:] = 0
        self.attitude_error_integral[:] = 0
        
        # Очищаем историю
        self.command_history.clear()
        self.state_history.clear()
        
        # Очищаем очередь команд
        while not self.command_queue.empty():
            await self.command_queue.get()
            
        self.logger.info("Agent reset completed")
        
    async def run_control_loop(self) -> None:
        """
        Запускает основной цикл управления агентом.
        """
        self.logger.info("Starting agent control loop")
        try:
            while self.running:
                # Обрабатываем команды из очереди
                if not self.command_queue.empty():
                    thrust, torque = await self.command_queue.get()
                    self.command_history.append((time.time(), thrust, torque))
                    
                # Короткая пауза для снижения нагрузки
                await asyncio.sleep(0.01)
                
        except Exception as e:
            self.logger.error(f"Control loop error: {str(e)}")
            await self._engage_emergency_stop()
        finally:
            self.logger.info("Control loop stopped")

    def _evaluate_overall_status(
        self,
        integrity_percentage: float,
        total_battery_percentage: float,
        core_temperature_c: float,
        stress_level: float,
    ) -> Tuple[str, str]:
        """
        Оценивает общее состояние агента на основе различных показателей.
        Возвращает кортеж (текущий статус, требуемое действие).

        Args:
            integrity_percentage (float): Процент целостности каркаса.
            total_battery_percentage (float): Общий процент заряда батарей.
            core_temperature_c (float): Температура ядра бота в °C.
            stress_level (float): Уровень структурного стресса (0.0-1.0).

        Returns:
            Tuple[str, str]: Кортеж (строка статуса, строка требуемого действия).
                             Возможные действия: "EMERGENCY_STOP", "REDUCE_ACTIVITY", "NONE".
        """
        # Проверка целостности
        if integrity_percentage < config.CRITICAL_INTEGRITY_THRESHOLD:
            self.logger.critical(
                f"Critical integrity: {integrity_percentage:.1f}%! "
                "Initiating emergency stop."
            )
            return "CRITICAL_INTEGRITY", "EMERGENCY_STOP"

        # Проверка батареи
        if total_battery_percentage < config.LOW_BATTERY_THRESHOLD:
            self.logger.warning(
                f"Low battery: {total_battery_percentage:.1f}%! "
                "Prioritizing power conservation."
            )
            return "LOW_BATTERY", "REDUCE_ACTIVITY"

        # Проверка температуры
        temp_status, temp_action = self._evaluate_temperature_status(core_temperature_c)
        if temp_action != "NONE":
            return temp_status, temp_action

        # Проверка уровня стресса
        if stress_level > config.STRESS_DAMAGE_THRESHOLD:
            self.logger.warning(
                f"High stress level: {stress_level:.2f}. "
                "Reducing activity to mitigate damage."
            )
            return "HIGH_STRESS", "REDUCE_ACTIVITY"

        return "NOMINAL", "NONE"

    def _evaluate_temperature_status(
        self, core_temperature_c: float
    ) -> Tuple[str, str]:
        """
        Оценивает температурное состояние агента.
        Возвращает кортеж (статус температуры, требуемое действие).

        Args:
            core_temperature_c (float): Температура ядра бота в °C.

        Returns:
            Tuple[str, str]: Кортеж (строка статуса температуры, строка требуемого действия).
        """
        if core_temperature_c > config.OVERHEAT_THRESHOLD_C:
            self.logger.critical(
                f"Core temperature {core_temperature_c:.1f}°C "
                "is too high (Threshold: {config.OVERHEAT_THRESHOLD_C}°C)! Initiating emergency stop."
            )
            return "OVERHEAT_CRITICAL", "EMERGENCY_STOP"
        elif core_temperature_c < config.FREEZE_THRESHOLD_C:
            self.logger.critical(
                f"Core temperature {core_temperature_c:.1f}°C "
                "is too low (Threshold: {config.FREEZE_THRESHOLD_C}°C)! Initiating emergency stop."
            )
            return "FREEZING_CRITICAL", "EMERGENCY_STOP"
        elif core_temperature_c > config.WARNING_TEMPERATURE_C:
            self.logger.warning(
                f"Core temperature {core_temperature_c:.1f}°C "
                "is high (Warning: {config.WARNING_TEMPERATURE_C}°C). Reducing activity."
            )
            return "HIGH_TEMP_WARNING", "REDUCE_ACTIVITY"
        elif core_temperature_c < config.WARNING_TEMPERATURE_C_LOW:
            self.logger.warning(
                f"Core temperature {core_temperature_c:.1f}°C "
                "is low (Warning: {config.WARNING_TEMPERATURE_C_LOW}°C). Reducing activity."
            )
            return "LOW_TEMP_WARNING", "REDUCE_ACTIVITY"

        return "TEMP_NOMINAL", "NONE"

    def _navigate_to_target(
        self,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        current_forward_vector: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вспомогательная функция для логики навигации к цели.

        Args:
            current_position (np.ndarray): Текущая позиция бота.
            current_velocity (np.ndarray): Текущая скорость бота.
            current_forward_vector (np.ndarray): Текущий вектор "вперед" бота.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Кортеж (вектор тяги, вектор крутящего момента).
        """
        target_thrust = np.zeros(3)
        target_torque = np.zeros(3)

        vector_to_target = self.target - current_position
        distance_to_target = np.linalg.norm(vector_to_target)

        if distance_to_target > config.TARGET_THRESHOLD:
            direction_to_target = vector_to_target / distance_to_target

            # Простая логика тяги: чем дальше, тем сильнее тяга.
            # Учитываем текущую скорость, чтобы не проскочить цель (простое демпфирование).
            # Коэффициент для уменьшения тяги при приближении и/или движении к цели.
            # Чем выше скорость по направлению к цели, тем меньше тяга.
            velocity_component_towards_target = np.dot(
                current_velocity, direction_to_target
            )

            # Коэффициент демпфирования, чтобы уменьшить тягу при приближении или высокой скорости к цели
            # Это очень упрощенная модель, можно использовать PID-регулятор для более точного управления.
            damping_factor = 1.0 - np.clip(
                velocity_component_towards_target / config.MAX_VELOCITY, 0, 1
            )
            thrust_magnitude = (
                config.THRUSTER_FORCE
                * damping_factor
                * np.clip(distance_to_target / 10.0, 0.1, 1.0)
            )  # Масштабируем по расстоянию

            target_thrust = direction_to_target * thrust_magnitude

            # Коррекция ориентации: выравниваемся по направлению к цели
            dot_product = np.dot(current_forward_vector, direction_to_target)
            # Если угол между "вперед" и направлением к цели значителен (например, > 5-10 градусов)
            # cos(5 deg) ~ 0.996, cos(10 deg) ~ 0.984
            if dot_product < 0.98:  # Порог для начала коррекции ориентации
                # Векторное произведение даст ось вращения
                rotation_axis = np.cross(current_forward_vector, direction_to_target)
                # Нормализуем ось вращения
                norm_rotation_axis = np.linalg.norm(rotation_axis)
                if norm_rotation_axis > 1e-6:  # Избегаем деления на ноль
                    rotation_axis /= norm_rotation_axis

                # Величина крутящего момента может зависеть от угла отклонения
                # acos(dot_product) - угол отклонения
                angle_diff = np.arccos(np.clip(dot_product, -1.0, 1.0))
                torque_magnitude = config.TORQUE_MAX * np.clip(
                    angle_diff / (np.pi / 4), 0.1, 1.0
                )  # Масштабируем по углу

                target_torque = rotation_axis * torque_magnitude
        else:
            self.logger.info(
                f"Target {self.target.tolist()} reached at {current_position.tolist()}. "
                "Setting new random target."
            )
            self.set_random_target()  # Установка новой случайной цели
            # Останавливаем движение, когда цель достигнута
            target_thrust = (
                -current_velocity * self.physics_obj.mass / self.decision_interval
            )  # Попытка остановить бота
            target_torque = (
                -self.physics_obj.get_state().angular_velocity
                * np.diag(self.physics_obj.inertia_tensor)
                / self.decision_interval
            )

        # Проверка, находится ли цель в пределах BOUNDS (уже есть в set_random_target)
        # но можно добавить и здесь для текущей цели, если она была установлена извне
        bounds = config.BOUNDS
        if not (
            bounds["x"][0] <= self.target[0] <= bounds["x"][1]
            and bounds["y"][0] <= self.target[1] <= bounds["y"][1]
            and bounds["z"][0] <= self.target[2] <= bounds["z"][1]
        ):
            self.logger.warning(
                f"Current target {self.target.tolist()} is outside bounds. "
                "Setting new random target."
            )
            self.set_random_target()

        return target_thrust, target_torque

    def set_random_target(self) -> None:
        """
        Устанавливает новую случайную целевую позицию в пределах,
        определенных в `config.BOUNDS`.
        """
        bounds = config.BOUNDS
        new_target_x = np.random.uniform(bounds["x"][0], bounds["x"][1])
        new_target_y = np.random.uniform(bounds["y"][0], bounds["y"][1])
        new_target_z = np.random.uniform(bounds["z"][0], bounds["z"][1])
        self.target = np.array([new_target_x, new_target_y, new_target_z])
        self.logger.info(f"New random target set: {self.target.tolist()}")

    def set_emergency_stop(self, status: bool) -> None:
        """
        Устанавливает или снимает аварийную остановку агента.

        Args:
            status (bool): True для активации аварийной остановки, False для деактивации.
        """
        self.emergency_stop = status
        if status:
            self.logger.critical("EMERGENCY STOP ACTIVATED!")
        else:
            self.logger.info("Emergency stop deactivated.")

    async def update(self, sensor_data: Dict[str, Any]) -> None:
        """
        Асинхронно обновляет состояние агента на основе данных сенсоров.

        Args:
            sensor_data: Словарь с данными от всех сенсоров
        """
        try:
            # Сохраняем текущее состояние в историю
            current_time = time.time()
            self.state_history.append((current_time, sensor_data))
            
            # Проверяем критические показатели
            if await self._check_critical_conditions(sensor_data):
                return
                
            # В неавтономном режиме только мониторим состояние
            if not self.autonomous:
                return
                
            # Ограничиваем частоту принятия решений
            if current_time - self.last_decision_time < self.decision_interval:
                return
                
            # Принимаем решение и отправляем команды
            thrust, torque = await self._make_decision(sensor_data)
            await self.command_queue.put((thrust, torque))
            
            # Сохраняем команду в историю
            self.command_history.append((current_time, thrust, torque))
            
            # Обновляем время последнего решения
            self.last_decision_time = current_time
            
            # Оповещаем о принятии решения
            self.decision_event.set()
            
        except Exception as e:
            self.logger.error(f"Agent update failed: {str(e)}")
            await self._engage_emergency_stop()
            
    async def _check_critical_conditions(self, sensor_data: Dict[str, Any]) -> bool:
        """
        Проверяет критические условия системы.
        
        Returns:
            bool: True если обнаружены критические условия
        """
        try:
            # Проверка батареи
            power_data = sensor_data.get("power", {})
            total_battery = power_data.get("total_battery_percentage", 0.0)
            if total_battery < config.CRITICAL_BATTERY_LEVEL:
                self.logger.warning(f"Critical battery level: {total_battery}%")
                await self._engage_emergency_stop()
                return True
                
            # Проверка температуры
            thermal_data = sensor_data.get("thermal", {})
            core_temp = thermal_data.get("core_temperature_c", 0.0)
            if core_temp > config.CRITICAL_TEMPERATURE_C:
                self.logger.warning(f"Critical temperature: {core_temp}°C")
                await self._engage_emergency_stop()
                return True
                
            # Проверка целостности каркаса
            frame_data = sensor_data.get("frame", {})
            integrity = frame_data.get("integrity_percentage", 100.0)
            if integrity < config.CRITICAL_INTEGRITY_LEVEL:
                self.logger.warning(f"Critical frame integrity: {integrity}%")
                await self._engage_emergency_stop()
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking critical conditions: {str(e)}")
            await self._engage_emergency_stop()
            return True
            
    async def _make_decision(self, sensor_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Принимает решение о действиях на основе данных сенсоров.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Векторы тяги и момента
        """
        if self.emergency_stop:
            return np.zeros(3), np.zeros(3)
            
        try:
            # Извлекаем навигационные данные
            nav_data = sensor_data.get("navigation", {})
            current_position = np.array(nav_data.get("position", [0, 0, 0]))
            current_velocity = np.array(nav_data.get("velocity", [0, 0, 0]))
            current_angular_velocity = np.array(nav_data.get("angular_velocity", [0, 0, 0]))
            
            # Расчет ошибок позиции и скорости
            position_error = self.target - current_position
            # Мы используем только position_error для PID-регулятора
            
            # PID для позиции
            self.position_error_integral += position_error * self.decision_interval
            position_derivative = -current_velocity
            
            # Расчет управляющей силы через PID
            thrust = (
                self.pid_gains["position"]["P"] * position_error +
                self.pid_gains["position"]["I"] * self.position_error_integral +
                self.pid_gains["position"]["D"] * position_derivative
            )
            
            # Ограничение максимальной тяги
            thrust_magnitude = np.linalg.norm(thrust)
            if thrust_magnitude > self.max_thrust:
                thrust = thrust * (self.max_thrust / thrust_magnitude)
                
            # PID для угловой ориентации
            self.attitude_error_integral += current_angular_velocity * self.decision_interval
            attitude_derivative = current_angular_velocity
            
            # Расчет момента через PID
            torque = -(
                self.pid_gains["attitude"]["P"] * current_angular_velocity +
                self.pid_gains["attitude"]["I"] * self.attitude_error_integral +
                self.pid_gains["attitude"]["D"] * attitude_derivative
            )
            
            # Ограничение максимального момента
            torque_magnitude = np.linalg.norm(torque)
            if torque_magnitude > self.max_torque:
                torque = torque * (self.max_torque / torque_magnitude)
                
            return thrust, torque
            
        except Exception as e:
            self.logger.error(f"Decision making failed: {str(e)}")
            return np.zeros(3), np.zeros(3)

    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус агента.
        
        Returns:
            Dict[str, Any]: Словарь с данными статуса
        """
        return {
            "target": self.target.tolist(),
            "autonomous": self.autonomous,
            "emergency_stop": self.emergency_stop,
            "last_decision_time": self.last_decision_time,
            "command_count": len(self.command_history)
        }

    def process(self, dt: float, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной метод обработки состояния и принятия решений.
        
        Args:
            dt: Время прошедшее с последнего обновления
            sensor_data: Данные от всех сенсоров
            
        Returns:
            Dict[str, Any]: Команды для физического движка
        """
        current_time = time.time()
        if current_time - self.last_decision_time < self.decision_interval:
            return {"thrust": np.zeros(3), "torque": np.zeros(3)}

        if not self.autonomous or self.emergency_stop:
            return {"thrust": np.zeros(3), "torque": np.zeros(3)}
            
        # Анализ состояния и принятие решения
        status, action = self._evaluate_overall_status(
            sensor_data.get("frame", {}).get("integrity_percentage", 100.0),
            sensor_data.get("power", {}).get("total_battery_percentage", 100.0),
            sensor_data.get("thermal", {}).get("core_temperature_c", config.INITIAL_CORE_TEMPERATURE_C),
            sensor_data.get("frame", {}).get("stress_level", 0.0)
        )
        
        if action == "EMERGENCY_STOP":
            self.set_emergency_stop(True)
            return {"thrust": np.zeros(3), "torque": np.zeros(3)}
            
        # Навигация к цели
        pos = sensor_data.get("physics", {}).get("position", np.zeros(3))
        vel = sensor_data.get("physics", {}).get("velocity", np.zeros(3))
        fwd = sensor_data.get("physics", {}).get("forward", np.array([1, 0, 0]))
        
        thrust, torque = self._navigate_to_target(pos, vel, fwd)
        
        self.last_decision_time = current_time
        return {"thrust": thrust, "torque": torque}
