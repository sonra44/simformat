# qik_os.py (Исправленная версия)

import asyncio
import heapq
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Union
import logging
import weakref
import numpy as np
from collections import defaultdict, deque
from statistics import mean

# Использование нашего централизованного логгера
from logger import Logger


class TaskState(Enum):
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    CANCELLED = auto()


@dataclass(order=True)
class Task:
    """Представление задачи для выполнения QikOS."""
    priority: int
    task_id: str = field(compare=False)
    command: str = field(compare=False)
    status: TaskState = field(default=TaskState.QUEUED, compare=False)
    submit_time: float = field(default_factory=time.time, compare=False)
    start_time: Optional[float] = field(default=None, compare=False)
    end_time: Optional[float] = field(default=None, compare=False)
    result: Any = field(default=None, compare=False)
    error_message: Optional[str] = field(default=None, compare=False)
    # Дополнительные поля для отслеживания (например, кто отправил задачу)
    source: str = field(default="UNKNOWN", compare=False)
    # Ограничение по времени выполнения задачи (в секундах)
    timeout: Optional[float] = field(default=None, compare=False)


class BotInterface(ABC):
    """
    Абстрактный базовый класс, определяющий интерфейс для взаимодействия
    QikOS с физическими и сенсорными компонентами бота.
    Реализуется конкретным классом (например, QikiBotInterface),
    который предоставляет доступ к реальной симуляции или оборудованию.
    """

    # --- Общее состояние и телеметрия ---
    @abstractmethod
    def get_current_sim_time(self) -> float:
        """Возвращает текущее время симуляции."""
        pass

    @abstractmethod
    def get_bot_position(self) -> np.ndarray:
        """Возвращает текущую позицию бота (X, Y, Z) в метрах."""
        pass

    @abstractmethod
    def get_bot_velocity(self) -> np.ndarray:
        """Возвращает текущую скорость бота (Vx, Vy, Vz) в м/с."""
        pass

    @abstractmethod
    def get_bot_acceleration(self) -> np.ndarray:
        """Возвращает текущее ускорение бота (Ax, Ay, Az) в м/с^2."""
        pass

    @abstractmethod
    def get_bot_orientation_quat(self) -> np.ndarray:
        """Возвращает текущую ориентацию бота в виде кватерниона (x, y, z, w)."""
        pass

    @abstractmethod
    def get_bot_angular_velocity(self) -> np.ndarray:
        """Возвращает текущую угловую скорость бота (Wx, Wy, Wz) в рад/с."""
        pass

    @abstractmethod
    def get_target_position(self) -> np.ndarray:
        """Возвращает текущую целевую позицию для автономного движения."""
        pass

    @abstractmethod
    def get_aggregated_sensor_data(self) -> Dict[str, Any]:
        """Возвращает агрегированные данные со всех сенсоров."""
        pass

    @abstractmethod
    def get_aggregated_system_status(self) -> Dict[str, Any]:
        """Возвращает агрегированный статус всех бортовых систем (питание, тепло, каркас)."""
        pass

    # --- Действия (команды) ---
    @abstractmethod
    async def apply_thruster_force(self, force: np.ndarray) -> bool:
        """
        Применяет силу тяги к боту.

        Args:
            force (np.ndarray): Вектор силы [Fx, Fy, Fz] в Ньютонах.

        Returns:
            bool: True, если команда успешно принята; False в противном случае.
        """
        pass

    @abstractmethod
    async def apply_torque(self, torque: np.ndarray) -> bool:
        """
        Применяет крутящий момент к боту.

        Args:
            torque (np.ndarray): Вектор крутящего момента [Tx, Ty, Tz] в Н*м.

        Returns:
            bool: True, если команда успешно принята; False в противном случае.
        """
        pass

    @abstractmethod
    async def set_new_target(self, target_position: np.ndarray) -> bool:
        """
        Устанавливает новую целевую позицию для бота.

        Args:
            target_position (np.ndarray): Новая целевая позиция [X, Y, Z] в метрах.

        Returns:
            bool: True, если команда успешно принята; False в противном случае.
        """
        pass

    @abstractmethod
    async def activate_emergency_stop(self) -> bool:
        """
        Активирует аварийную остановку бота, отключая все двигатели.

        Returns:
            bool: True, если команда успешно принята.
        """
        pass

    @abstractmethod
    async def deactivate_emergency_stop(self) -> bool:
        """
        Деактивирует аварийную остановку бота, позволяя возобновить работу.

        Returns:
            bool: True, если команда успешно принята.
        """
        pass

    @abstractmethod
    async def set_autonomous_mode(self, enable: bool) -> bool:
        """
        Включает или выключает автономный режим агента.

        Args:
            enable (bool): True для включения, False для выключения.

        Returns:
            bool: True, если команда успешно принята.
        """
        pass

    @abstractmethod
    def get_solar_irradiance(self) -> float:
        """Получает текущую солнечную иррадиацию в местоположении бота."""
        pass

    @abstractmethod
    def get_gravity_vector(self) -> np.ndarray:
        """Возвращает текущий вектор гравитации."""
        pass

    @abstractmethod
    def get_forward_vector(self) -> np.ndarray:
        """Возвращает текущий "передний" вектор ориентации бота."""
        pass


class MockBotInterface(BotInterface):
    """
    Мок-реализация BotInterface для тестирования QikOS без полной симуляции.
    """

    def __init__(self):
        self._position = np.array([0.0, 0.0, 0.0])
        self._velocity = np.array([0.0, 0.0, 0.0])
        self._acceleration = np.array([0.0, 0.0, 0.0])
        self._orientation_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._angular_velocity = np.array([0.0, 0.0, 0.0])
        self._target_position = np.array([10.0, 10.0, 10.0])
        self._emergency_stop_active = False
        self._autonomous_mode_enabled = True
        self._solar_irradiance = 1000.0  # W/m^2
        self._gravity_vector = np.array([0.0, 0.0, -9.81])
        self._forward_vector = np.array([1.0, 0.0, 0.0])
        self._current_sim_time = 0.0

    def get_current_sim_time(self) -> float:
        return self._current_sim_time

    def get_bot_position(self) -> np.ndarray:
        return self._position

    def get_bot_velocity(self) -> np.ndarray:
        return self._velocity

    def get_bot_acceleration(self) -> np.ndarray:
        return self._acceleration

    def get_bot_orientation_quat(self) -> np.ndarray:
        return self._orientation_quat

    def get_bot_angular_velocity(self) -> np.ndarray:
        return self._angular_velocity

    def get_target_position(self) -> np.ndarray:
        return self._target_position

    def get_aggregated_sensor_data(self) -> Dict[str, Any]:
        return {
            "power": {"total_battery_percentage": 95.0, "solar_irradiance_w_m2": self._solar_irradiance},
            "thermal": {"core_temperature_c": 25.0, "status_message": "NOMINAL"},
            "frame": {"structural_integrity": 99.0, "stress_level": 0.1},
        }

    def get_aggregated_system_status(self) -> Dict[str, Any]:
        return {
            "power": {"total_battery_percentage": 95.0, "status": "NOMINAL"},
            "thermal": {"core_temperature_c": 25.0, "status_message": "NOMINAL"},
            "frame": {"structural_integrity": 99.0, "status_message": "NOMINAL"},
        }

    async def apply_thruster_force(self, force: np.ndarray) -> bool:
        if self._emergency_stop_active:
            return False
        # self._acceleration += force / 1000.0 # Примерное влияние на мок-состояние
        return True

    async def apply_torque(self, torque: np.ndarray) -> bool:
        if self._emergency_stop_active:
            return False
        # self._angular_velocity += torque / 100.0 # Примерное влияние
        return True

    async def set_new_target(self, target_position: np.ndarray) -> bool:
        self._target_position = target_position
        return True

    async def activate_emergency_stop(self) -> bool:
        self._emergency_stop_active = True
        return True

    async def deactivate_emergency_stop(self) -> bool:
        self._emergency_stop_active = False
        return True

    async def set_autonomous_mode(self, enable: bool) -> bool:
        self._autonomous_mode_enabled = enable
        return True

    def get_solar_irradiance(self) -> float:
        return self._solar_irradiance

    def get_gravity_vector(self) -> np.ndarray:
        return self._gravity_vector

    def get_forward_vector(self) -> np.ndarray:
        return self._forward_vector

    # Методы для управления состоянием мока в тестах
    def _set_position(self, pos: np.ndarray):
        self._position = pos

    def _set_sim_time(self, t: float):
        self._current_sim_time = t


class Kernel:
    """
    Ядро QikOS управляет задачами, их приоритетами и выполнением.
    Использует очередь с приоритетами для эффективной обработки задач.
    """

    def __init__(self, bot_interface: BotInterface):
        self.logger = Logger.get_logger("qik_os.kernel")
        self.task_queue: List[Task] = []  # Приоритетная очередь задач (heapq)
        self.active_tasks: Dict[str, asyncio.Task] = {}  # Задачи, которые сейчас выполняются
        self.completed_tasks: Deque[Task] = deque(maxlen=100)  # История завершенных задач
        self.failed_tasks: Deque[Task] = deque(maxlen=50)  # История проваленных задач
        self.bot_interface: BotInterface = bot_interface  # Ссылка на интерфейс бота

        # Статистика ядра
        self.stats = {
            "total_processed": 0,
            "completed": 0,
            "failed": 0,
            "timeout": 0,
            "cancelled": 0,
        }
        self.processing_times: Deque[float] = deque(maxlen=200) # История времени обработки задач
        self.logger.info("QikOS Kernel initialized.")

    def submit_task(
        self, command: str, priority: int = 5, timeout: Optional[float] = None, source: str = "EXTERNAL"
    ) -> str:
        """
        Добавляет новую задачу в очередь ядра.

        Args:
            command (str): Команда для выполнения (например, "move 10 20 30").
            priority (int): Приоритет задачи (чем меньше число, тем выше приоритет, 1 - самый высокий).
            timeout (Optional[float]): Максимальное время выполнения задачи в секундах.
            source (str): Источник задачи (например, "AUTONOMOUS", "MANUAL", "EXTERNAL").

        Returns:
            str: Уникальный ID задачи.
        """
        task_id = str(uuid.uuid4())
        new_task = Task(
            priority=priority,
            task_id=task_id,
            command=command,
            timeout=timeout,
            source=source
        )
        heapq.heappush(self.task_queue, new_task)
        self.logger.info(
            f"Task '{command}' (ID: {task_id[:8]}...) submitted with priority {priority} from {source}."
        )
        return task_id

    async def _execute_command(self, task: Task) -> Any:
        """
        Внутренний метод для выполнения команды через BotInterface.
        Разбирает команду и вызывает соответствующий метод BotInterface.
        """
        self.logger.debug(f"Executing task ID: {task.task_id[:8]}..., Command: '{task.command}'")
        parts = task.command.split(" ")
        action = parts[0].lower()
        args = parts[1:]

        try:
            result = False
            if action == "move_to":
                if len(args) == 3:
                    target_pos = np.array([float(args[0]), float(args[1]), float(args[2])])
                    result = await self.bot_interface.set_new_target(target_pos)
                else:
                    raise ValueError("move_to requires 3 arguments: x y z")
            elif action == "apply_force":
                if len(args) == 3:
                    force_vec = np.array([float(args[0]), float(args[1]), float(args[2])])
                    result = await self.bot_interface.apply_thruster_force(force_vec)
                else:
                    raise ValueError("apply_force requires 3 arguments: fx fy fz")
            elif action == "apply_torque":
                if len(args) == 3:
                    torque_vec = np.array([float(args[0]), float(args[1]), float(args[2])])
                    result = await self.bot_interface.apply_torque(torque_vec)
                else:
                    raise ValueError("apply_torque requires 3 arguments: tx ty tz")
            elif action == "emergency_stop":
                result = await self.bot_interface.activate_emergency_stop()
            elif action == "resume_operations":
                result = await self.bot_interface.deactivate_emergency_stop()
            elif action == "set_autonomous_mode":
                if len(args) == 1 and args[0].lower() in ["true", "false"]:
                    enable = args[0].lower() == "true"
                    result = await self.bot_interface.set_autonomous_mode(enable)
                else:
                    raise ValueError("set_autonomous_mode requires 1 argument: true/false")
            elif action == "get_position":
                result = self.bot_interface.get_bot_position().tolist()
                self.logger.info(f"Bot position: {result}")
            elif action == "get_sensor_data":
                result = self.bot_interface.get_aggregated_sensor_data()
                self.logger.info(f"Sensor data: {result}")
            elif action == "get_system_status":
                result = self.bot_interface.get_aggregated_system_status()
                self.logger.info(f"System status: {result}")
            else:
                raise ValueError(f"Unknown command: {action}")

            return result

        except Exception as e:
            self.logger.error(f"Error executing command '{task.command}' (ID: {task.task_id[:8]}...): {e}", exc_info=True)
            raise

    async def process_tasks(self, max_iterations: int = 1) -> None:
        """
        Обрабатывает задачи из очереди.
        Вызывается в каждом цикле QikOS.
        """
        processed_count = 0
        while self.task_queue and processed_count < max_iterations:
            task = heapq.heappop(self.task_queue)
            self.logger.debug(f"Picked task ID: {task.task_id[:8]}..., Command: '{task.command}'")

            if task.status != TaskState.QUEUED:
                self.logger.warning(f"Task ID: {task.task_id[:8]}... already processed or cancelled. Skipping.")
                continue

            task.status = TaskState.RUNNING
            task.start_time = time.time()
            self.stats["total_processed"] += 1

            try:
                # Создаем задачу asyncio и добавляем ее в активные
                # Если у задачи есть таймаут, оборачиваем в asyncio.wait_for
                if task.timeout is not None:
                    coro = asyncio.wait_for(self._execute_command(task), timeout=task.timeout)
                else:
                    coro = self._execute_command(task)

                async_task = asyncio.create_task(coro)
                self.active_tasks[task.task_id] = async_task

                try:
                    task.result = await async_task
                    task.status = TaskState.COMPLETED
                    self.stats["completed"] += 1
                    self.completed_tasks.append(task)
                    self.logger.info(f"Task ID: {task.task_id[:8]}... completed. Result: {task.result}")
                except asyncio.TimeoutError:
                    task.status = TaskState.TIMEOUT
                    task.error_message = f"Task timed out after {task.timeout} seconds."
                    self.stats["timeout"] += 1
                    self.failed_tasks.append(task)
                    self.logger.warning(f"Task ID: {task.task_id[:8]}... timed out.")
                except asyncio.CancelledError:
                    task.status = TaskState.CANCELLED
                    task.error_message = "Task was cancelled."
                    self.stats["cancelled"] += 1
                    self.failed_tasks.append(task) # Отмененные тоже считаем в "неудавшиеся" для общей статистики
                    self.logger.info(f"Task ID: {task.task_id[:8]}... cancelled.")
                except Exception as e:
                    task.status = TaskState.FAILED
                    task.error_message = str(e)
                    self.stats["failed"] += 1
                    self.failed_tasks.append(task)
                    self.logger.error(f"Task ID: {task.task_id[:8]}... failed with error: {e}")
            finally:
                task.end_time = time.time()
                if task.start_time:
                    self.processing_times.append(task.end_time - task.start_time)
                # Удаляем задачу из активных, так как она завершена (успешно, с ошибкой, таймаут, отмена)
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                processed_count += 1

    async def cancel_task(self, task_id: str) -> bool:
        """Попытаться отменить активную или ожидающую задачу."""
        # Проверяем активные задачи
        if task_id in self.active_tasks:
            task_asyncio = self.active_tasks[task_id]
            if not task_asyncio.done():
                task_asyncio.cancel()
                self.logger.info(f"Attempting to cancel active task: {task_id[:8]}...")
                # Дать asyncio.CancelledError распространиться
                try:
                    await task_asyncio
                except asyncio.CancelledError:
                    pass # Ожидаемое исключение
                return True
            else:
                self.logger.warning(f"Task {task_id[:8]}... is already done, cannot cancel.")
                return False

        # Проверяем ожидающие задачи в очереди
        for i, task in enumerate(self.task_queue):
            if task.task_id == task_id:
                task.status = TaskState.CANCELLED
                task.end_time = time.time()
                task.error_message = "Task cancelled by request."
                self.stats["cancelled"] += 1
                self.failed_tasks.append(task) # Отмененные тоже в "неудавшиеся"
                # Удаляем из кучи, перестраивая её, если нужно
                # Простой способ - пометить как отмененную и не обрабатывать, когда она будет извлечена
                # Более сложный, но чистый - удалить и перестроить кучу:
                # self.task_queue.pop(i) # Нельзя просто pop из кучи
                # heapq.heapify(self.task_queue)
                self.logger.info(f"Queued task {task_id[:8]}... marked as CANCELLED.")
                return True

        self.logger.warning(f"Task {task_id[:8]}... not found in active or queued tasks.")
        return False


    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает текущий статус задачи по ее ID."""
        # Проверяем активные задачи
        if task_id in self.active_tasks:
            task = next((t for t in self.task_queue if t.task_id == task_id), None)
            if task: return task.status.name # Лучше возвращать объект Task
            # Если задача уже начала выполняться и извлечена из task_queue
            # (но все еще в active_tasks), то статус должен быть RUNNING
            return TaskState.RUNNING.name


        # Проверяем ожидающие задачи
        for task in self.task_queue:
            if task.task_id == task_id:
                return task.status.name

        # Проверяем завершенные задачи (успешные, проваленные, таймаут, отмененные)
        for task_list in [self.completed_tasks, self.failed_tasks]:
            for task in task_list:
                if task.task_id == task_id:
                    return task.status.name

        return None # Задача не найдена

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы ядра."""
        current_time = time.time()
        running_tasks_info = []
        for task_id, task_asyncio in self.active_tasks.items():
            task_obj = next((t for t in self.task_queue if t.task_id == task_id), None)
            if task_obj: # Если задача еще в очереди (только что начата)
                running_tasks_info.append({
                    "id": task_id,
                    "command": task_obj.command,
                    "status": task_obj.status.name,
                    "elapsed": current_time - task_obj.start_time if task_obj.start_time else 0,
                    "timeout": task_obj.timeout
                })
            else: # Если задача уже была извлечена из очереди
                # Нужно найти объект Task из self.completed_tasks или self.failed_tasks
                # Это немного усложняет, если мы не храним активные задачи как объекты Task
                # Пока что просто логируем их как "RUNNING"
                running_tasks_info.append({
                    "id": task_id,
                    "status": TaskState.RUNNING.name,
                    "elapsed": "N/A",
                    "timeout": "N/A"
                })


        avg_processing_time = mean(self.processing_times) if self.processing_times else 0.0

        stats = {
            "total_processed": self.stats["total_processed"],
            "completed": self.stats["completed"],
            "failed": self.stats["failed"],
            "timeout": self.stats["timeout"],
            "cancelled": self.stats["cancelled"],
            "pending_tasks": len(self.task_queue),
            "running_tasks_count": len(self.active_tasks),
            "running_tasks_details": running_tasks_info, # Детализация активных задач
            "avg_processing_time_s": avg_processing_time,
            "queue_depth": len(self.task_queue),
            "completed_history_count": len(self.completed_tasks),
            "failed_history_count": len(self.failed_tasks),
        }
        return stats


class Shell:
    """
    Оболочка QikOS предоставляет пользовательский интерфейс (или программный API)
    для отправки команд ядру и получения результатов.
    """

    def __init__(self, kernel: Kernel):
        self.logger = Logger.get_logger("qik_os.shell")
        self.kernel: Kernel = kernel
        self.command_history: List[Dict[str, Any]] = [] # История всех отправленных команд
        self.logger.info("QikOS Shell initialized.")

    def execute_command(self, command_string: str, priority: int = 5, timeout: Optional[float] = None) -> str:
        """
        Парсит и отправляет команду ядру.

        Args:
            command_string (str): Строка команды, введенная пользователем.
            priority (int): Приоритет команды.
            timeout (Optional[float]): Таймаут для выполнения команды.

        Returns:
            str: ID задачи, созданной для команды.
        """
        task_id = self.kernel.submit_task(command_string, priority, timeout, source="SHELL")
        self.command_history.append({"id": task_id, "command": command_string, "status": "QUEUED"})
        self.logger.info(f"Command '{command_string}' submitted via shell. Task ID: {task_id[:8]}...")
        return task_id

    async def get_command_result(self, task_id: str) -> Optional[Any]:
        """
        Ожидает и возвращает результат выполнения команды по ID задачи.
        """
        # Эта логика должна быть более сложной, чтобы не блокировать.
        # В реальной системе это может быть через механизм событий или пулинг.
        # Для простоты демо, можно использовать короткий цикл ожидания или Future.
        self.logger.debug(f"Waiting for result of task {task_id[:8]}...")
        task_status = self.kernel.get_task_status(task_id)
        while task_status in [TaskState.QUEUED.name, TaskState.RUNNING.name]:
            await asyncio.sleep(0.05) # Короткая пауза
            task_status = self.kernel.get_task_status(task_id)

        # Найдем задачу в истории завершенных/проваленных
        for task_list in [self.kernel.completed_tasks, self.kernel.failed_tasks]:
            for task in task_list:
                if task.task_id == task_id:
                    self._update_command_history_status(task_id, task.status.name)
                    return task.result if task.status == TaskState.COMPLETED else task.error_message
        return None # Задача не найдена или не завершена

    async def cancel_command(self, task_id: str) -> bool:
        """Попытаться отменить команду."""
        success = await self.kernel.cancel_task(task_id)
        if success:
            self._update_command_history_status(task_id, TaskState.CANCELLED.name)
        return success

    def _update_command_history_status(self, task_id: str, status: str) -> None:
        """Внутренний метод для обновления статуса команды в истории."""
        for entry in self.command_history:
            if entry["id"] == task_id:
                entry["status"] = status
                self.logger.debug(f"Updated status for task {task_id[:8]}... to {status}")
                break

    def get_history(self) -> List[Dict[str, Any]]:
        """Возвращает полную историю команд."""
        # Для полноты, обновляем статусы в истории перед возвратом
        for entry in self.command_history:
            task_id = entry["id"]
            kernel_status = self.kernel.get_task_status(task_id)
            if kernel_status and kernel_status != entry["status"]:
                entry["status"] = kernel_status
        return list(self.command_history)


class QikOS:
    """
    Основной класс операционной системы QIKI.
    Агрегирует ядро и оболочку, предоставляя высокоуровневый интерфейс.
    """

    def __init__(self, bot_interface: BotInterface):
        self.logger = Logger.get_logger("qik_os")
        self.kernel: Kernel = Kernel(bot_interface)
        self.shell: Shell = Shell(self.kernel)
        self._running: bool = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_task: Optional[asyncio.Task] = None
        self.logger.info("QikOS initialized.")

    async def start(self) -> None:
        """Запускает QikOS."""
        if self._running:
            self.logger.warning("QikOS is already running.")
            return

        self.logger.info("Starting QikOS...")
        self._running = True
        # asyncio.get_running_loop() должен вызываться из уже запущенного цикла
        # Если QikOS запускается из внешнего цикла, то _loop будет не None
        # Если QikOS запускает свой собственный цикл, то его нужно создать здесь
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self.logger.info("Created new asyncio event loop for QikOS.")

        self._main_task = self._loop.create_task(self._run_main_loop())
        self.logger.info("QikOS started.")

    async def _run_main_loop(self) -> None:
        """Основной асинхронный цикл работы QikOS."""
        self.logger.info("QikOS main loop started.")
        while self._running:
            try:
                await self.kernel.process_tasks()
                await asyncio.sleep(0.01)  # Короткая пауза для предотвращения блокировки
            except asyncio.CancelledError:
                self.logger.info("QikOS main loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in QikOS main loop: {e}", exc_info=True)
                # В случае критической ошибки можно попытаться остановить QikOS
                # await self.stop()
        self.logger.info("QikOS main loop finished.")

    async def stop(self) -> None:
        """Останавливает QikOS и отменяет все активные задачи."""
        if not self._running:
            self.logger.warning("QikOS is not running.")
            return

        self.logger.info("Stopping QikOS...")
        self._running = False

        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                self.logger.info("QikOS main task was cancelled successfully.")
            except Exception as e:
                self.logger.error(f"Error while waiting for QikOS main task to stop: {e}")
            finally:
                self._main_task = None

        # Отменяем все активные задачи ядра
        for task_id in list(self.kernel.active_tasks.keys()):
            await self.kernel.cancel_task(task_id)

        # Даем короткое время на завершение отмененных задач
        await asyncio.sleep(0.1)

        self.logger.info("QikOS stopped.")


# Демонстрационный запуск (для отладки QikOS как отдельного модуля)
async def qikos_demo():
    print("\n--- Starting QikOS Demo ---")
    Logger.get_logger().set_level(logging.INFO) # Устанавливаем уровень логирования для демо
    logger = Logger.get_logger("qikos_demo")
    logger.info("QikOS demo logger initialized.")

    mock_bot = MockBotInterface()
    qikos = QikOS(bot_interface=mock_bot)

    await qikos.start()

    logger.info("Submitting demo commands...")
    task_id_move = qikos.shell.execute_command("move_to 10 20 5", priority=1)
    task_id_force = qikos.shell.execute_command("apply_force 100 0 0", priority=2, timeout=0.1)
    task_id_get_pos = qikos.shell.execute_command("get_position", priority=3)
    task_id_sensor = qikos.shell.execute_command("get_sensor_data", priority=4)
    task_id_unknown = qikos.shell.execute_command("unknown_command arg1", priority=5)

    # Демонстрация обновления времени для мок-интерфейса
    for i in range(10):
        mock_bot._set_sim_time(i * 0.1) # Имитируем течение времени
        mock_bot._set_position(np.array([1.0 * i, 2.0 * i, 0.0])) # Имитируем движение
        await asyncio.sleep(0.05) # Небольшая пауза для симуляции времени

    logger.info("Attempting to get results...")
    result_move = await qikos.shell.get_command_result(task_id_move)
    logger.info(f"Result of move_to: {result_move}")

    result_force = await qikos.shell.get_command_result(task_id_force)
    logger.info(f"Result of apply_force: {result_force}")

    result_get_pos = await qikos.shell.get_command_result(task_id_get_pos)
    logger.info(f"Result of get_position: {result_get_pos}")

    result_sensor = await qikos.shell.get_command_result(task_id_sensor)
    logger.info(f"Result of get_sensor_data: {result_sensor}")

    result_unknown = await qikos.shell.get_command_result(task_id_unknown)
    logger.info(f"Result of unknown_command: {result_unknown}")

    logger.info("QikOS Stats:")
    print(qikos.kernel.get_stats())

    logger.info("QikOS History:")
    for cmd in qikos.shell.get_history():
        print(f"  {cmd['command']} -> {cmd['status']}")

    await qikos.stop()

    # Дадим асинхронным задачам шанс завершиться
    await asyncio.sleep(0.2)

    stats = qikos.kernel.get_stats()
    print(f"\n=== Final Statistics (Demo) ===")
    print(f"Total processed: {stats['total_processed']}")
    print(f"Completed: {stats['completed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Timeout: {stats['timeout']}")
    print(f"Cancelled: {stats['cancelled']}")

    history = qikos.shell.get_history()
    print(f"\n=== Command History (Demo) ===")
    for i, cmd_info in enumerate(history):
        print(f"  {i+1}. ID: {cmd_info['id'][:8]}..., Command: '{cmd_info['command']}', Status: {cmd_info['status']}{f', Error: {cmd_info.get('error_message', '')}' if cmd_info.get('error_message') else ''}")

    print("\nQikOS demo finished.")


if __name__ == "__main__":
    # Для демонстрации
    # Создание необходимых директорий
    required_dirs = ["logs", "data"]
    for dirname in required_dirs:
        os.makedirs(dirname, exist_ok=True)

    # Инициализация нашего логгера для демо
    # В реальном main.py это будет сделано в QikiSimulation.__init__
    # Но для автономного запуска qik_os.py, это нужно здесь
    try:
        Logger.get_logger().set_level(logging.INFO) # Устанавливаем уровень логирования для всего приложения
        logging.getLogger("qikos_demo").info("Initialized main logger for QikOS demo.")
    except Exception as e:
        # Fallback для логирования, если инициализация логгера не удалась
        print(f"Failed to initialize logger for QikOS demo: {e}", file=sys.stderr)
        logging.basicConfig(level=logging.INFO) # Использовать базовый логгер Python

    try:
        asyncio.run(qikos_demo())
    except KeyboardInterrupt:
        print("\nQikOS demo interrupted by user.")
    except Exception as e:
        logging.getLogger("qikos_demo").critical(f"An unexpected error occurred during QikOS demo: {e}", exc_info=True)