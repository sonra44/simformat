import numpy as np
import time
import math
import random
import asyncio
from typing import Dict, Any, Optional, TYPE_CHECKING
from config import config
from logger import Logger

if TYPE_CHECKING:
    from physics import PhysicsObject

# Константы
DEFAULT_SOLAR_IRRADIANCE_W_M2: float = 1000.0  # Стандартное значение для ясного дня


class Environment:
    """
    Асинхронный класс Environment моделирует окружающую среду симуляции,
    включая гравитацию, солнечную радиацию, сопротивление среды и границы мира.
    """

    def __init__(self):
        """
        Инициализирует окружение с поддержкой асинхронных операций.
        """
        self.logger: Logger = Logger.get_logger("environment")
        self.gravity: np.ndarray = config.GRAVITY
        self.air_drag_coefficient: float = config.ENVIRONMENT_DRAG_COEFFICIENT
        self.bounds: dict = config.BOUNDS
        self.bot_radius: float = config.BOT_RADIUS
        
        # Время симуляции и параметры среды
        self.simulation_start_time = time.time()
        self.current_time = 0.0
        self._last_weather_update = 0.0
        self._current_weather_factor = 1.0
        
        # События для синхронизации
        self.update_event = asyncio.Event()
        self.weather_update_event = asyncio.Event()
        
        self.logger.info(
            f"Async Environment initialized with gravity {self.gravity.tolist()} "
            f"and bounds {self.bounds}, bot radius {self.bot_radius}m."
        )

    async def update(self, dt: float, physics_obj: Optional["PhysicsObject"] = None) -> None:
        """
        Асинхронно обновляет состояние окружения.
        
        Args:
            dt: Временной шаг в секундах
            physics_obj: Опциональный физический объект для взаимодействия
        """
        if dt <= 0:
            raise ValueError(f"Invalid time step: {dt}")
            
        try:
            # Обновляем время симуляции
            self.current_time += dt
            
            # Обновляем погодные условия каждые 5 минут симуляции
            if self.current_time - self._last_weather_update >= 300:
                await self._update_weather()
                self._last_weather_update = self.current_time
            
            # Если передан физический объект, обрабатываем взаимодействие
            if physics_obj is not None:
                await self.apply_environmental_forces(physics_obj)
                await self.enforce_boundaries(physics_obj)
                
            # Оповещаем об обновлении
            self.update_event.set()
            
        except Exception as e:
            self.logger.error(f"Error updating environment: {str(e)}")
            raise

    async def _update_weather(self) -> None:
        """
        Асинхронно обновляет погодные условия.
        Влияет на солнечную радиацию и другие параметры среды.
        """
        # Используем текущее время для seed
        random.seed(int(self.current_time))
        
        # Обновляем погодный фактор
        self._current_weather_factor = random.uniform(0.7, 1.0)
        
        # Оповещаем об изменении погоды
        self.weather_update_event.set()
        
        self.logger.debug(f"Weather updated: factor = {self._current_weather_factor:.2f}")

    def get_solar_irradiance(self, position=None) -> float:
        """
        Возвращает текущую солнечную радиацию с учетом времени суток и погоды.
        
        Args:
            position: Позиция объекта (опционально)
            
        Returns:
            float: Значение солнечной радиации в Вт/м²
        """
        # Расчет времени суток (24-часовой цикл)
        day_progress = (self.current_time % 86400) / 86400
        
        # Базовая интенсивность по синусоиде
        base_intensity = max(0, math.sin(day_progress * 2 * math.pi))
        
        # Применяем текущий погодный фактор
        irradiance = DEFAULT_SOLAR_IRRADIANCE_W_M2 * base_intensity * self._current_weather_factor
        
        self.logger.debug(f"Solar irradiance: {irradiance:.1f} W/m² (time: {day_progress:.2f}, weather: {self._current_weather_factor:.2f})")
        return irradiance

    async def apply_environmental_forces(self, physics_obj: "PhysicsObject") -> None:
        """
        Асинхронно применяет силы окружающей среды к физическому объекту.
        
        Args:
            physics_obj: Физический объект для применения сил
        """
        state = physics_obj.get_state()
        
        # Применяем гравитацию
        await physics_obj.apply_force(self.gravity * physics_obj.mass)
        
        # Применяем сопротивление воздуха
        if np.any(state.velocity):
            drag_force = -self.air_drag_coefficient * state.velocity * np.abs(state.velocity)
            await physics_obj.apply_force(drag_force)

    async def enforce_boundaries(self, physics_obj: "PhysicsObject") -> None:
        """
        Асинхронно проверяет и корректирует положение объекта относительно границ мира.
        
        Args:
            physics_obj: Физический объект для проверки
        """
        state = physics_obj.get_state()
        position = state.position
        velocity = state.velocity
        
        # Проверяем каждую координату
        for i, (coord, vel) in enumerate(zip(position, velocity)):
            min_bound = self.bounds[f"min_{chr(120+i)}"]  # x, y, z -> min_x, min_y, min_z
            max_bound = self.bounds[f"max_{chr(120+i)}"]
            
            if coord - self.bot_radius < min_bound:
                # Объект достиг нижней границы
                position[i] = min_bound + self.bot_radius
                if velocity[i] < 0:
                    velocity[i] = 0  # Останавливаем движение в этом направлении
            elif coord + self.bot_radius > max_bound:
                # Объект достиг верхней границы
                position[i] = max_bound - self.bot_radius
                if velocity[i] > 0:
                    velocity[i] = 0  # Останавливаем движение в этом направлении
        
        # Применяем обновленное состояние
        await physics_obj.wait_for_update()  # Ждем завершения текущего физического обновления
        physics_obj.state.position = position
        physics_obj.state.velocity = velocity

    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус окружающей среды.
        
        Returns:
            Dict с параметрами среды
        """
        return {
            "time": self.current_time,
            "day_progress": (self.current_time % 86400) / 86400,
            "solar_irradiance": self.get_solar_irradiance(),
            "weather_factor": self._current_weather_factor,
            "gravity": self.gravity.tolist(),
            "air_drag": self.air_drag_coefficient
        }

    async def reset(self) -> None:
        """
        Асинхронно сбрасывает состояние окружения к начальным значениям.
        """
        self.current_time = 0.0
        self._last_weather_update = 0.0
        self._current_weather_factor = 1.0
        self.simulation_start_time = time.time()
        
        # Оповещаем об обновлении
        self.update_event.set()
        self.weather_update_event.set()
        
        self.logger.info("Environment reset to initial state")
