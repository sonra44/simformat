# hardware/power_systems.py
import numpy as np
from config import config  # Импортируем глобальный экземпляр конфигурации
from logger import Logger  # Корректно: из корневой папки
from typing import Dict, Any, TYPE_CHECKING


if TYPE_CHECKING:
    # Для подсказок типов, если PowerSystems будет импортироваться куда-то еще.
    pass


class PowerSensors:
    """
    Моделирует датчики энергетической системы, предоставляя информацию
    о состоянии батарей, генерации энергии и текущем потреблении.
    """

    def __init__(self, power_systems_instance: "PowerSystems"):
        """
        Инициализирует датчики энергетической системы.

        Args:
            power_systems_instance (PowerSystems): Экземпляр PowerSystems,
                                                   к которому привязаны датчики.
        """
        self.logger: Logger = Logger.get_logger("power_sensors")
        self.power_systems: "PowerSystems" = power_systems_instance
        self.main_battery_percentage: float = 100.0
        self.reserve_battery_percentage: float = 100.0
        self.solar_panel_output_w: float = 0.0
        self.current_draw_w: float = 0.0
        self.logger.info("PowerSensors initialized.")

    def update(self, solar_power_generated_w: float, current_draw_w: float) -> None:
        """
        Обновляет показания датчиков на основе сгенерированной и потребленной мощности.

        Args:
            solar_power_generated_w (float): Мощность, генерируемая солнечными панелями (Вт).
            current_draw_w (float): Текущее потребление мощности (Вт).
        """
        self.solar_panel_output_w = solar_power_generated_w
        self.current_draw_w = current_draw_w

        # Обновление процентов заряда батарей (для отображения, реальное управление в PowerSystems)
        # Это упрощенное представление, обычно проценты берутся напрямую из состояния батарей в PowerSystems
        total_charge_wh = self.power_systems.main_battery_charge_wh + self.power_systems.reserve_battery_charge_wh
        total_capacity_wh = config.MAIN_BATTERY_CAPACITY_WH + config.RESERVE_BATTERY_CAPACITY_WH
        if total_capacity_wh > 0:
            self.main_battery_percentage = (self.power_systems.main_battery_charge_wh / config.MAIN_BATTERY_CAPACITY_WH) * 100
            self.reserve_battery_percentage = (self.power_systems.reserve_battery_charge_wh / config.RESERVE_BATTERY_CAPACITY_WH) * 100
        else:
            self.main_battery_percentage = 0.0
            self.reserve_battery_percentage = 0.0

        # self.logger.debug(
        #     f"PowerSensors updated. Solar: {self.solar_panel_output_w:.1f}W, "
        #     f"Draw: {self.current_draw_w:.1f}W, Main Bat: {self.main_battery_percentage:.1f}%"
        # )

    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущие показания датчиков энергетической системы.

        Returns:
            Dict[str, Any]: Словарь с мощностью солнечных панелей, текущим потреблением,
                            и процентами заряда батарей.
        """
        return {
            "solar_panel_output_w": self.solar_panel_output_w,
            "current_draw_w": self.current_draw_w,
            "main_battery_percentage": self.main_battery_percentage,
            "reserve_battery_percentage": self.reserve_battery_percentage,
        }


class PowerSystems:
    """
    Класс PowerSystems моделирует энергетическую систему бота QIKI.
    Отвечает за генерацию энергии (солнечные панели), хранение (батареи)
    и распределение энергии, а также за мониторинг общего потребления.
    """

    def __init__(self, name: str = "QIKI_PowerSystems"):
        """
        Инициализирует энергетическую систему.

        Args:
            name (str, optional): Имя системы. По умолчанию "QIKI_PowerSystems".
        """
        self.logger: Logger = Logger.get_logger("power_systems")
        self.name: str = name

        # Емкость батарей в Ватт-часах (Wh)
        self.main_battery_charge_wh: float = config.MAIN_BATTERY_CAPACITY_WH
        self.reserve_battery_charge_wh: float = config.RESERVE_BATTERY_CAPACITY_WH

        # Инициализация датчиков энергетической системы
        self.power_sensors: PowerSensors = PowerSensors(self)

        self.solar_power_generated_w: float = 0.0
        self.total_power_draw_w: float = 0.0
        self.status_message: str = "NOMINAL"

        self.logger.info(
            f"{self.name} initialized. "
            f"Main Battery: {self.main_battery_charge_wh:.1f}Wh, "
            f"Reserve Battery: {self.reserve_battery_charge_wh:.1f}Wh."
        )

    def get_total_mass(self) -> float:
        """
        Возвращает общую массу энергетической системы (батареи + панели).

        Returns:
            float: Общая масса в кг.
        """
        return config.MAIN_BATTERY_MASS_KG + config.RESERVE_BATTERY_MASS_KG + config.SOLAR_PANEL_MASS_KG

    def update(self, dt: float, solar_irradiance_w_m2: float, system_power_draw_w: float) -> None:
        """
        Обновляет состояние энергетической системы за временной шаг dt.

        Args:
            dt (float): Временной шаг симуляции (секунды).
            solar_irradiance_w_m2 (float): Текущая солнечная иррадиация (Вт/м^2).
            system_power_draw_w (float): Общее текущее потребление мощности системой (Вт).
        """
        try:
            # Базовое потребление энергии системами
            base_power_draw = (
                config.POWER_CONSUMPTION_COMPUTER_W +
                config.POWER_CONSUMPTION_SENSORS_W +
                config.POWER_CONSUMPTION_IDLE_W
            )
            
            # Общее потребление с учетом КПД разряда
            total_power_draw = (base_power_draw + system_power_draw_w) / config.BATTERY_DISCHARGE_EFFICIENCY
            
            # Рассчитываем генерацию от солнечных панелей и сохраняем для телеметрии
            raw_solar_power = self._calculate_solar_power(solar_irradiance_w_m2)
            self.solar_power_generated_w = raw_solar_power
            
            # Часть энергии преобразуется в тепло (будет учтено в thermal_system)
            solar_power = raw_solar_power * (1.0 - config.SOLAR_PANEL_THERMAL_FACTOR)
            
            # Полезная энергия с учетом КПД заряда
            usable_solar_power = solar_power * config.BATTERY_CHARGE_EFFICIENCY
            
            # Сохраняем текущее потребление для телеметрии
            self.total_power_draw_w = total_power_draw
            
            # Рассчитываем реальный баланс энергии (в ватт-часах)
            energy_delta = (usable_solar_power - total_power_draw) * (dt / 3600.0) # переводим в ватт-часы
            
            if energy_delta >= 0:
                # При избытке энергии заряжаем сначала основную батарею
                remaining_main_capacity = config.MAIN_BATTERY_CAPACITY_WH - self.main_battery_charge_wh
                main_battery_charge = min(energy_delta, remaining_main_capacity)
                self.main_battery_charge_wh += main_battery_charge
                
                # Излишки направляем в резервную батарею
                remaining_energy = energy_delta - main_battery_charge
                if remaining_energy > 0:
                    remaining_reserve_capacity = config.RESERVE_BATTERY_CAPACITY_WH - self.reserve_battery_charge_wh
                    reserve_battery_charge = min(remaining_energy, remaining_reserve_capacity)
                    self.reserve_battery_charge_wh += reserve_battery_charge
            else:
                # При недостатке энергии разряжаем сначала основную батарею
                energy_needed = abs(energy_delta)
                main_battery_discharge = min(self.main_battery_charge_wh, energy_needed)
                self.main_battery_charge_wh -= main_battery_discharge
                
                # Если нужно больше энергии, используем резервную батарею
                remaining_energy_needed = energy_needed - main_battery_discharge
                if remaining_energy_needed > 0:
                    reserve_battery_discharge = min(self.reserve_battery_charge_wh, remaining_energy_needed)
                    self.reserve_battery_charge_wh -= reserve_battery_discharge
            
            # Обновляем датчики
            self.power_sensors.update(solar_power, system_power_draw_w)
            
        except Exception as e:
            self.logger.error(f"Power systems update failed: {str(e)}")
            # Активируем аварийный режим
            self.emergency_mode = True

    def _calculate_solar_power(self, solar_irradiance_w_m2: float) -> float:
        """
        Рассчитывает мощность, генерируемую солнечными панелями.

        Args:
            solar_irradiance_w_m2 (float): Солнечная иррадиация в Вт/м².

        Returns:
            float: Сгенерированная мощность в Ваттах.
        """
        return (
            solar_irradiance_w_m2 * config.SOLAR_PANEL_AREA_M2 * config.SOLAR_PANEL_EFFICIENCY
        )

    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус энергетической системы.

        Returns:
            Dict[str, Any]: Словарь с зарядом основной/резервной батареи,
                            мощностью солнечных панелей, текущим потреблением
                            и сообщением о статусе.
        """
        main_perc = (self.main_battery_charge_wh / config.MAIN_BATTERY_CAPACITY_WH) * 100 if config.MAIN_BATTERY_CAPACITY_WH > 0 else 0
        reserve_perc = (self.reserve_battery_charge_wh / config.RESERVE_BATTERY_CAPACITY_WH) * 100 if config.RESERVE_BATTERY_CAPACITY_WH > 0 else 0

        total_charge_wh = self.main_battery_charge_wh + self.reserve_battery_charge_wh
        total_capacity_wh = config.MAIN_BATTERY_CAPACITY_WH + config.RESERVE_BATTERY_CAPACITY_WH
        total_battery_percentage = (
            (total_charge_wh / total_capacity_wh) * 100 if total_capacity_wh > 0 else 0
        )

        status_msg = "NOMINAL"
        if self.total_power_draw_w > config.OVERLOAD_THRESHOLD_W:
            status_msg = "OVERLOAD"
            self.logger.warning(
                f"Power system overload! Draw: {self.total_power_draw_w:.1f}W, Threshold: {config.OVERLOAD_THRESHOLD_W:.1f}W"
            )
        elif (
            total_battery_percentage < config.LOW_BATTERY_THRESHOLD
        ):  # Используем общий порог
            status_msg = "LOW BATTERY"
            self.logger.warning(f"Low battery! Total: {total_battery_percentage:.1f}%, Main: {main_perc:.1f}%, Reserve: {reserve_perc:.1f}%")
        elif (
            total_battery_percentage < config.WARNING_BATTERY_THRESHOLD
        ):  # Используем общий порог
            status_msg = "BATTERY WARNING"
            self.logger.info(f"Battery warning. Total: {total_battery_percentage:.1f}%, Main: {main_perc:.1f}%, Reserve: {reserve_perc:.1f}%")
        else:
            if self.status_message != "NOMINAL": # Логируем возврат в норму
                self.logger.info(f"Power system returned to NOMINAL. Total battery: {total_battery_percentage:.1f}%")
            status_msg = "NOMINAL" # Обновляем статус после всех проверок


        return {
            "name": self.name,
            "total_mass_kg": self.get_total_mass(),
            "main_battery_charge_wh": self.main_battery_charge_wh,
            "reserve_battery_charge_wh": self.reserve_battery_charge_wh,
            "main_battery_percentage": main_perc,
            "reserve_battery_percentage": reserve_perc,
            "total_battery_percentage": total_battery_percentage,
            "solar_panel_output_w": self.solar_power_generated_w,
            "total_power_draw_w": self.total_power_draw_w,
            "status_message": status_msg,
        }
    
    def get_total_battery_percentage(self) -> float:
        """
        Возвращает общий процент заряда батарей.
        
        Returns:
            float: Процент заряда (0-100)
        """
        total_charge_wh = self.main_battery_charge_wh + self.reserve_battery_charge_wh
        total_capacity_wh = config.MAIN_BATTERY_CAPACITY_WH + config.RESERVE_BATTERY_CAPACITY_WH
        return (total_charge_wh / total_capacity_wh) * 100 if total_capacity_wh > 0 else 0
    
    def get_power_consumption(self) -> float:
        """
        Возвращает текущее потребление энергии в ваттах.
        
        Returns:
            float: Потребление энергии в ваттах
        """
        return self.total_power_draw_w
    
    def get_solar_power(self) -> float:
        """
        Возвращает текущую мощность солнечных панелей в ваттах.
        
        Returns:
            float: Мощность солнечных панелей в ваттах
        """
        return self.solar_power_generated_w
    
    def get_charging_state(self) -> bool:
        """
        Возвращает статус зарядки батареи.
        
        Returns:
            bool: True если батарея заряжается
        """
        return self.solar_power_generated_w > self.total_power_draw_w