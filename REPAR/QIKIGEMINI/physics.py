# physics.py

import numpy as np
import asyncio
from typing import Optional, Dict, Any, Tuple
from logger import Logger
from dataclasses import dataclass, field
from collections import deque

# Проверяем доступность SciPy
try:
    from scipy.spatial.transform import Rotation as ScipyRotation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    ScipyRotation = None


# --- Определение класса DummyRotation перед блоком try/except ---
class DummyRotation:
    """
    Класс-заглушка для Rotation, если scipy.spatial.transform недоступен.
    Обеспечивает базовую совместимость, но не выполняет реальных вращений
    с полной математической точностью. Используется для сохранения работоспособности
    кода без полной функциональности вращения, если библиотека SciPy отсутствует.
    Для векторных преобразований используется упрощенная логика,
    что может привести к неточкам в сложных вращениях.
    """

    def __init__(self, quat=None):
        # quat: (x, y, z, w)
        if quat is None:
            self._quat = np.array([0.0, 0.0, 0.0, 1.0])  # Единичный кватернион (без вращения)
        else:
            self._quat = np.asarray(quat, dtype=float)
            if self._quat.shape != (4,):
                raise ValueError("Quaternion must be a 4-element array.")
            self._quat /= np.linalg.norm(self._quat) # Нормализация

    @staticmethod
    def from_euler(seq, angles, degrees=False):
        # Очень упрощенная заглушка, которая не делает реальных преобразований
        # Для реальной работы нужна SciPy
        Logger.get_logger("physics").warning(
            "DummyRotation: from_euler called, but SciPy is not available. "
            "Rotation will not be correctly applied."
        )
        return DummyRotation() # Всегда возвращаем identity

    def apply(self, vectors, inverse=False):
        # Для заглушки просто возвращаем векторы как есть, или инвертируем, если нужно
        # Это ОЧЕНЬ упрощенно и не является корректным вращением.
        Logger.get_logger("physics").warning(
            "DummyRotation: apply called, but SciPy is not available. "
            "Vectors will not be correctly rotated."
        )
        if inverse:
            return -np.asarray(vectors, dtype=float) # Просто инвертируем для демонстрации
        return np.asarray(vectors, dtype=float)

    def as_quat(self):
        # Возвращаем кватернион заглушки
        return self._quat

    def inv(self):
        # Возвращаем копию самого себя, т.к. для заглушки нет инверсии
        return DummyRotation(self._quat * np.array([1, 1, 1, -1])) # Упрощенная инверсия кватерниона

    def __mul__(self, other):
        # Для заглушки, умножение всегда приводит к идентичности
        Logger.get_logger("physics").warning(
            "DummyRotation: multiplication called, but SciPy is not available. "
            "Resulting rotation might be incorrect."
        )
        return DummyRotation()

# --- Инициализация класса Rotation для поддержки SciPy или использования заглушки ---
try:
    from scipy.spatial.transform import Rotation as ScipyRotation
    SCIPY_AVAILABLE = True
    Logger.get_logger("physics").info("SciPy Rotation available. Full rotational physics enabled.")
except ImportError:
    ScipyRotation = None
    SCIPY_AVAILABLE = False
    Logger.get_logger("physics").warning(
        "Module scipy.spatial.transform.Rotation not found. "
        "Using a DummyRotation class. Rotational physics will be basic and approximated. "
        "Consider installing SciPy for accurate rotational simulations (e.g., 'pip install scipy')."
    )

# --- Конец блока инициализации Rotation ---


@dataclass
class State:
    """
    Класс State хранит текущее физическое состояние объекта:
    позицию, скорость, ускорение, ориентацию (кватернион),
    угловую скорость и угловое ускорение.
    Все состояния представлены как массивы numpy для эффективных вычислений.
    """
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))  # [м]
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))  # [м/с]
    acceleration: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))  # [м/с^2]
    # Определяем тип как Any для совместимости с обеими реализациями
    orientation: Any = field(default_factory=lambda: (ScipyRotation.from_quat([0,0,0,1]) if SCIPY_AVAILABLE else DummyRotation()))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))  # [рад/с]
    angular_acceleration: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))  # [рад/с^2]

    def copy(self) -> "State":
        """Создает полную копию состояния."""
        return State(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            orientation=self.orientation,  # Ориентация уже неизменяемая
            angular_velocity=self.angular_velocity.copy(),
            angular_acceleration=self.angular_acceleration.copy()
        )

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Возвращает текущее состояние в виде словаря
        
        Returns:
            Dict с векторами состояния
        """
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'acceleration': self.acceleration.copy(),
            'orientation': self.orientation.as_quat() if SCIPY_AVAILABLE else np.array([0, 0, 0, 1]),
            'angular_velocity': self.angular_velocity.copy(),
            'angular_acceleration': self.angular_acceleration.copy()
        }
        
    def set_state(self, state_dict: Dict[str, np.ndarray]) -> None:
        """
        Устанавливает состояние из словаря
        
        Args:
            state_dict: Словарь с векторами состояния
        """
        self.position = state_dict['position'].copy()
        self.velocity = state_dict['velocity'].copy()
        self.acceleration = state_dict['acceleration'].copy()
        if SCIPY_AVAILABLE:
            self.orientation = ScipyRotation.from_quat(state_dict['orientation'])
        self.angular_velocity = state_dict['angular_velocity'].copy()
        self.angular_acceleration = state_dict['angular_acceleration'].copy()
        
    def copy_with(self, **kwargs) -> 'State':
        """
        Создает копию состояния с возможностью переопределения параметров
        
        Args:
            **kwargs: Параметры для переопределения
            
        Returns:
            Новый объект State
        """
        state_dict = self.get_state()
        state_dict.update(kwargs)
        new_state = State()
        new_state.set_state(state_dict)
        return new_state


class PhysicsObject:
    """Физический объект в симуляции с асинхронным управлением"""
    
    def __init__(self, name: str, mass: float, initial_state: Optional[State] = None, inertia: float = 1.0):
        """
        Инициализирует физический объект
        
        Args:
            name: Имя объекта
            mass: Масса в килограммах
            initial_state: Начальное состояние (позиция, скорость)
            inertia: Момент инерции объекта (кг*м^2)
        """
        self.name = name
        self.mass = mass
        self.inertia = inertia
        self.state = initial_state or State()
        
        # Текущие силы и моменты
        self.forces = np.zeros(3)  # Суммарный вектор сил
        self.torques = np.zeros(3)  # Суммарный вектор моментов
        
        # Очереди команд
        self.force_queue = asyncio.Queue()  # Очередь команд для сил
        self.torque_queue = asyncio.Queue()  # Очередь команд для моментов
        
        # События для синхронизации
        self.update_event = asyncio.Event()  # Событие обновления физики
        self.command_event = asyncio.Event()  # Событие получения новой команды
        
        # История состояний для визуализации
        self.state_history = deque(maxlen=100)  # Последние 100 состояний
        
        self.logger = Logger.get_logger(f"physics.{name}")
        self.logger.info(f"PhysicsObject '{name}' initialized with mass {mass} kg, inertia {inertia} kg*m^2")
        
    async def apply_force(self, force: np.ndarray) -> None:
        """
        Асинхронно применяет силу к объекту
        
        Args:
            force: Вектор силы в Ньютонах
        """
        await self.force_queue.put(force)
        self.command_event.set()
        
    async def apply_torque(self, torque: np.ndarray) -> None:
        """
        Асинхронно применяет крутящий момент к объекту
        
        Args:
            torque: Вектор момента в Н*м
        """
        await self.torque_queue.put(torque)
        self.command_event.set()
        
    async def _process_commands(self) -> None:
        """Обрабатывает все накопленные команды из очередей"""
        # Обработка сил
        while not self.force_queue.empty():
            force = await self.force_queue.get()
            self.forces += force
            
        # Обработка моментов
        while not self.torque_queue.empty():
            torque = await self.torque_queue.get()
            self.torques += torque
            
    async def update(self, dt: float) -> None:
        """
        Асинхронно обновляет физическое состояние объекта
        
        Args:
            dt: Временной шаг в секундах
        """
        if dt <= 0:
            self.logger.warning(f"Invalid dt value: {dt}")
            return
            
        try:
            # Обрабатываем входящие команды
            await self._process_commands()
            
            # Обновляем линейное движение
            self.state.position += self.state.velocity * dt
            acceleration = self.forces / self.mass
            self.state.velocity += acceleration * dt
            self.state.acceleration = acceleration
            
            # Обновляем вращательное движение
            angular_acceleration = self.torques / self.inertia
            self.state.angular_velocity += angular_acceleration * dt
            self.state.angular_acceleration = angular_acceleration
            
            # Обновляем ориентацию
            if SCIPY_AVAILABLE:
                delta_angle = self.state.angular_velocity * dt
                delta_rotation = ScipyRotation.from_rotvec(delta_angle)
                self.state.orientation = delta_rotation * self.state.orientation
            else:
                self.logger.warning("Rotation update skipped: SciPy not available")
                
            # Сохраняем состояние в историю
            self.state_history.append(self.state.copy_with())
            
            # Сбрасываем силы и моменты после обновления
            self.forces[:] = 0
            self.torques[:] = 0
            
            # Оповещаем о завершении обновления
            self.update_event.set()
            
        except Exception as e:
            self.logger.error(f"Physics update failed: {str(e)}")
            raise
            
    def get_state(self) -> State:
        """Возвращает текущее состояние объекта"""
        return self.state
        
    def get_state_history(self) -> Tuple[State, ...]:
        """Возвращает историю состояний объекта"""
        return tuple(self.state_history)
        
    async def wait_for_update(self) -> None:
        """Ожидает завершения следующего физического обновления"""
        self.update_event.clear()
        await self.update_event.wait()
        
    async def wait_for_command(self) -> None:
        """Ожидает получения следующей команды"""
        self.command_event.clear()
        await self.command_event.wait()
        
    async def reset(self) -> None:
        """Асинхронно сбрасывает состояние объекта"""
        self.state = State()
        self.forces[:] = 0
        self.torques[:] = 0
        self.state_history.clear()
        
        # Очищаем очереди
        while not self.force_queue.empty():
            await self.force_queue.get()
        while not self.torque_queue.empty():
            await self.torque_queue.get()
            
        self.logger.info(f"PhysicsObject '{self.name}' reset to initial state")
        
    def get_forward_vector(self) -> np.ndarray:
        """Возвращает вектор направления вперед на основе текущей ориентации"""
        try:
            if SCIPY_AVAILABLE:
                # Если SciPy доступен, используем настоящие кватернионы
                # [0, 0, 1] - базовый вектор "вперед" в исходной системе координат
                forward = np.array([0.0, 0.0, 1.0])
                return self.state.orientation.apply(forward)
            else:
                # Если SciPy недоступен, просто возвращаем базовый вектор
                self.logger.warning("Forward vector calculation simplified: SciPy not available")
                return np.array([0.0, 0.0, 1.0])
        except Exception as e:
            self.logger.error(f"Error getting forward vector: {str(e)}")
            return np.array([0.0, 0.0, 1.0])  # Возвращаем безопасное значение по умолчанию
    
    @property
    def net_force(self) -> np.ndarray:
        """Текущий суммарный вектор сил"""
        return self.forces