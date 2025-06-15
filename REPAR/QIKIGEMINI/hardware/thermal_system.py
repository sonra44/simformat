# hardware/thermal_system.py
import numpy as np
from config import config  # Импортируем глобальный экземпляр конфигурации
from logger import Logger # Корректно: из корневой папки
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    # Для подсказок типов, если ThermalSystem будет импортироваться куда-то еще.
    pass


class ThermalSystem:
    """
    Класс, моделирующий термическую систему QIKI.
    Отвечает за расчет и управление температурой основных компонентов,
    включая ядро и радиаторы.
    """

    def __init__(self, name: str = "QIKI_ThermalSystem"):
        """
        Инициализирует термическую систему.

        Args:
            name (str, optional): Имя термической системы.
                                  По умолчанию "QIKI_ThermalSystem".
        """
        self.logger: Logger = Logger.get_logger("thermal_system")
        self.name: str = name

        # Начальные температуры из конфигурации
        self.core_temperature_c: float = config.INITIAL_CORE_TEMPERATURE_C
        self.radiator_temperature_c: float = config.INITIAL_RADIATOR_TEMPERATURE_C

        self.status_message: str = "NOMINAL"
        self.logger.info(
            f"{self.name} initialized. "
            f"Initial Core Temp: {self.core_temperature_c:.1f}°C, "
            f"Initial Radiator Temp: {self.radiator_temperature_c:.1f}°C."
        )

    def update(self, dt: float, total_heat_generated_w: float) -> None:
        """
        Обновляет температуру системы за временной шаг dt.

        Args:
            dt (float): Временной шаг симуляции (секунды).
            total_heat_generated_w (float): Общая тепловая мощность, генерируемая
                                            в системе (Вт), например, от электроники.
        """
        # 1. Расчет теплообмена между ядром и радиатором
        # Простая модель теплопередачи: Q = h * A * dT
        # Где h - коэффициент теплопередачи, A - площадь, dT - разница температур
        # Используем CONFIG.HEAT_TRANSFER_COEFFICIENT_CORE_RADIATOR
        heat_transferred_core_to_radiator_w = (
            config.HEAT_TRANSFER_COEFFICIENT_CORE_RADIATOR
            * (self.core_temperature_c - self.radiator_temperature_c)
        )

        # 2. Расчет теплопотерь радиатором в космос (излучение)
        # Закон Стефана-Больцмана: P = ε * σ * A * T^4
        # Температуры должны быть в Кельвинах (C + 273.15)
        radiator_temperature_k = self.radiator_temperature_c + 273.15
        heat_radiated_to_space_w = (
            config.EMISSIVITY
            * config.STEFAN_BOLTZMANN_CONSTANT
            * config.RADIATOR_SURFACE_AREA_M2
            * (radiator_temperature_k**4 - config.SPACE_TEMPERATURE_K**4) # Учитываем температуру пространства
        )
        # Убедимся, что тепло излучается, только если радиатор горячее космоса
        heat_radiated_to_space_w = max(0, heat_radiated_to_space_w)


        # 3. Теплообмен с окружающей средой
        ambient_heat_transfer_core = config.AMBIENT_HEAT_TRANSFER_COEFFICIENT * (config.AMBIENT_TEMPERATURE_C - self.core_temperature_c)
        ambient_heat_transfer_radiator = config.AMBIENT_HEAT_TRANSFER_COEFFICIENT * (config.AMBIENT_TEMPERATURE_C - self.radiator_temperature_c)

        # 4. Обновление температуры ядра
        # Учитываем генерацию тепла, теплопередачу радиатору и теплообмен с окружающей средой
        net_heat_core_w = total_heat_generated_w - heat_transferred_core_to_radiator_w + ambient_heat_transfer_core
        delta_core_temp_c = (net_heat_core_w / config.CORE_THERMAL_MASS_J_PER_C) * dt
        self.core_temperature_c += delta_core_temp_c

        # 5. Обновление температуры радиатора
        # Учитываем теплопередачу от ядра, излучение в космос и теплообмен с окружающей средой
        net_heat_radiator_w = heat_transferred_core_to_radiator_w - heat_radiated_to_space_w + ambient_heat_transfer_radiator
        delta_radiator_temp_c = (net_heat_radiator_w / config.RADIATOR_THERMAL_MASS_J_PER_C) * dt
        self.radiator_temperature_c += delta_radiator_temp_c

        # Ограничиваем температуры реалистичными пределами
        self.core_temperature_c = np.clip(
            self.core_temperature_c, 
            config.MIN_CORE_TEMPERATURE_C,
            config.OVERHEAT_THRESHOLD_C + 20.0  # Позволяем немного превысить порог перегрева
        )
        self.radiator_temperature_c = np.clip(
            self.radiator_temperature_c,
            config.MIN_RADIATOR_TEMPERATURE_C,
            config.OVERHEAT_THRESHOLD_C  # Радиатор не должен быть горячее порога перегрева
        )

        self.check_status() # Обновляем статус системы

        # self.logger.debug(
        #     f"ThermalSystem updated. Core Temp: {self.core_temperature_c:.1f}°C, "
        #     f"Radiator Temp: {self.radiator_temperature_c:.1f}°C."
        # )

    def check_status(self) -> None:
        """
        Проверяет текущую температуру ядра и обновляет статус системы.
        """
        if self.core_temperature_c >= config.OVERHEAT_THRESHOLD_C:
            self.status_message = "CRITICAL OVERHEAT"
            self.logger.critical(
                f"Core temperature CRITICAL: {self.core_temperature_c:.1f}°C "
                f"(Threshold: {config.OVERHEAT_THRESHOLD_C}°C)"
            )
        elif self.core_temperature_c >= config.WARNING_TEMPERATURE_C:
            self.status_message = "OVERHEAT WARNING"
            self.logger.warning(
                f"Core temperature WARNING: {self.core_temperature_c:.1f}°C "
                f"(Warning Threshold: {config.WARNING_TEMPERATURE_C}°C)"
            )
        elif self.core_temperature_c <= config.FREEZE_THRESHOLD_C:
            self.status_message = "CRITICAL FREEZING"
            self.logger.critical(
                f"Core temperature CRITICAL: {self.core_temperature_c:.1f}°C "
                f"(Threshold: {config.FREEZE_THRESHOLD_C}°C)"
            )
        elif self.core_temperature_c <= config.WARNING_TEMPERATURE_C_LOW:
            self.status_message = "LOW TEMP WARNING"
            self.logger.warning(
                f"Core temperature LOW: {self.core_temperature_c:.1f}°C "
                f"(Warning Threshold: {config.WARNING_TEMPERATURE_C_LOW}°C)"
            )
        else:
            if self.status_message != "NOMINAL":  # Логируем возврат в норму
                self.logger.info(
                    f"Core temperature returned to NOMINAL: {self.core_temperature_c:.1f}°C"
                )
            self.status_message = "NOMINAL"

    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус термической системы.

        Returns:
            Dict[str, Any]: Словарь с текущей температурой ядра, радиатора
                            и сообщением о статусе.
        """
        return {
            "name": self.name,
            "core_temperature_c": self.core_temperature_c,
            "radiator_temperature_c": self.radiator_temperature_c,
            "status_message": self.status_message,
        }

    def get_power_consumption(self) -> float:
        """
        Рассчитывает текущее энергопотребление термической системы.
        Включает энергозатраты на активное охлаждение, если оно требуется.

        Returns:
            float: Текущее энергопотребление в ваттах
        """
        # Базовое потребление на мониторинг температуры
        base_power_w = config.THERMAL_SYSTEM_BASE_POWER_W
        
        # Дополнительное потребление на активное охлаждение
        # Пропорционально разнице между текущей температурой и порогом перегрева
        if self.core_temperature_c > config.THERMAL_WARNING_THRESHOLD_C:
            temp_delta = self.core_temperature_c - config.THERMAL_WARNING_THRESHOLD_C
            cooling_power = config.COOLING_POWER_PER_DEGREE_W * temp_delta
        else:
            cooling_power = 0.0
            
        return base_power_w + cooling_power