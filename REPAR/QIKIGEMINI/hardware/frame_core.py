# hardware/frame_core.py
import numpy as np
from config import config  # Импортируем глобальный экземпляр конфигурации
from logger import Logger # Корректно: из корневой папки
from typing import Dict, TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Это для подсказок типов, чтобы избежать циклического импорта,
    # если FrameCore будет импортироваться куда-то еще, что маловероятно для этого файла.
    pass


class StructuralSensors:
    """
    Имитирует структурные датчики бота, измеряя деформацию,
    вибрацию и общую целостность каркаса.
    """

    def __init__(self, frame_core: "FrameCore"):
        """
        Инициализирует структурные датчики.

        Args:
            frame_core (FrameCore): Экземпляр FrameCore, к которому привязаны датчики.
        """
        self.logger: Logger = Logger.get_logger("structural_sensors")
        self.frame_core: "FrameCore" = frame_core
        self.stress_level: float = 0.0  # Уровень деформации (0.0 - 1.0)
        self.vibration_level: float = 0.0  # Уровень вибрации (0.0 - 1.0)
        self.integrity_percentage: float = 100.0  # Процент целостности (100% - цел)
        self.logger.info("StructuralSensors initialized.")

    def update(
        self, current_acceleration: np.ndarray, current_angular_acceleration: np.ndarray
    ) -> None:
        """
        Обновляет показания структурных датчиков на основе физических воздействий.

        Args:
            current_acceleration (np.ndarray): Текущее линейное ускорение бота (м/с^2).
            current_angular_acceleration (np.ndarray): Текущее угловое ускорение бота (рад/с^2).
        """
        # Моделирование влияния ускорений на целостность и стресс
        # Уровень стресса увеличивается с ростом ускорения
        linear_stress = np.linalg.norm(current_acceleration) * config.STRUCTURAL_STRESS_FACTOR
        angular_stress = np.linalg.norm(current_angular_acceleration) * config.STRUCTURAL_ANGULAR_STRESS_FACTOR

        self.stress_level = min(1.0, self.stress_level + linear_stress + angular_stress)
        self.vibration_level = min(1.0, self.vibration_level + linear_stress * 0.5) # Вибрация чуть меньше стресса

        # Целостность уменьшается, если стресс превышает порог
        if self.stress_level > config.STRUCTURAL_INTEGRITY_THRESHOLD:
            degradation = (self.stress_level - config.STRUCTURAL_INTEGRITY_THRESHOLD) * config.STRUCTURAL_DEGRADATION_RATE
            self.integrity_percentage = max(0.0, self.integrity_percentage - degradation)

        # self.logger.debug(
        #     f"StructuralSensors updated. Stress: {self.stress_level:.2f}, "
        #     f"Integrity: {self.integrity_percentage:.2f}%"
        # )

    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущие показания структурных датчиков.

        Returns:
            Dict[str, Any]: Словарь с уровнем стресса, вибрации и процентом целостности.
        """
        return {
            "stress_level": self.stress_level,
            "vibration_level": self.vibration_level,
            "integrity_percentage": self.integrity_percentage,
        }


class FrameCore:
    """Основной каркас робота QIKI"""
    
    def __init__(self, name: str = "QIKI_FrameCore", base_mass: float = 5.0):
        """
        Инициализирует каркас робота
        
        Args:
            name: Имя компонента
            base_mass: Базовая масса каркаса в кг
        """
        self.logger: Logger = Logger.get_logger("frame_core")
        self.name: str = name
        self.base_mass: float = base_mass

        # Инициализация структурных датчиков, привязанных к этому каркасу
        self.structural_sensors: StructuralSensors = StructuralSensors(self)
        self.structural_integrity: float = (
            self.structural_sensors.integrity_percentage
        )  # Процент целостности каркаса

        # Базовое энергопотребление для сенсоров и систем каркаса
        self.base_power_consumption = 10.0  # Ватт
        
        self.logger.info(
            f"FrameCore '{self.name}' initialized with base mass {self.base_mass:.2f} kg."
        )

    def get_power_consumption(self) -> float:
        """
        Возвращает текущее энергопотребление каркаса
        
        Returns:
            float: Энергопотребление в ваттах
        """
        # В будущем можно добавить зависимость от нагрузки
        return self.base_power_consumption

    def get_total_mass(self) -> float:
        """
        Возвращает полную массу каркаса
        
        Returns:
            float: Масса в кг
        """
        return self.base_mass

    def update(
        self, current_acceleration: np.ndarray, current_angular_acceleration: np.ndarray
    ) -> None:
        """
        Обновляет состояние каркаса и его датчиков на основе текущих ускорений.

        Args:
            current_acceleration (np.ndarray): Текущее линейное ускорение бота (м/с^2).
            current_angular_acceleration (np.ndarray): Текущее угловое ускорение бота (рад/с^2).
        """
        self.structural_sensors.update(
            current_acceleration, current_angular_acceleration
        )
        self.structural_integrity = self.structural_sensors.integrity_percentage

    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус каркаса и его датчиков.

        Returns:
            Dict[str, any]: Словарь, содержащий имя, массу, структурную целостность
                            и данные от структурных датчиков.
        """
        return {
            "name": self.name,
            "mass": self.base_mass,
            "structural_integrity": self.structural_integrity,
            "sensors_data": self.structural_sensors.get_status(),
        }