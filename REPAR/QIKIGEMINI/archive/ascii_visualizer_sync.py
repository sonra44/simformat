import os
import sys
import time
import numpy as np
from typing import Dict, Any

class AsciiVisualizer:
    def __init__(self, width: int = 80, height: int = 40):  # Увеличиваем высоту для большего количества информации
        self.width = width
        self.height = height
        self.buffer = []
        self.start_time = time.time()
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.clear_screen()
        
    def clear_screen(self):
        """Очищает экран и буфер"""
        self.buffer = []
        # Только перемещаем курсор вверх вместо очистки
        if os.name != 'nt':  # Для Unix-подобных систем
            print("\033[H\033[J", end='')  # Очистка через ANSI escape codes
        else:
            os.system('cls')  # Для Windows
            
    def _add_to_buffer(self, text: str):
        """Добавляет текст в буфер"""
        self.buffer.append(text)
        
    def _flush_buffer(self):
        """Выводит буфер на экран"""
        print('\n'.join(self.buffer))
        sys.stdout.flush()  # Принудительный вывод
        
    def update(self, state, sensors_data: Dict[str, Any], environment) -> None:
        """
        Обновляет ASCII визуализацию
        
        Args:
            state: Текущее состояние физического объекта
            sensors_data: Данные с сенсоров
            environment: Текущее состояние окружающей среды
        """
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.frame_count += 1
        fps = 1.0 / dt if dt > 0 else 0
        
        self.clear_screen()
        
        # Получаем данные состояния и сенсоров
        pos = state.position
        vel = state.velocity
        acc = state.acceleration if hasattr(state, 'acceleration') else np.zeros(3)
        ang_vel = state.angular_velocity if hasattr(state, 'angular_velocity') else np.zeros(3)
        orientation = state.orientation if hasattr(state, 'orientation') else np.array([0, 0, 0, 1])
        
        # Данные систем
        power_data = sensors_data.get("power", {})
        thermal_data = sensors_data.get("thermal", {})
        frame_data = sensors_data.get("frame", {})
        nav_data = sensors_data.get("navigation", {})
        
        # Основные показатели
        bat = power_data.get("total_battery_percentage", 0.0)
        bat_reserve = power_data.get("reserve_battery_percentage", 0.0)
        solar_power = power_data.get("solar_panel_output_w", 0.0)
        power_draw = power_data.get("current_draw_w", 0.0)
        
        temp = thermal_data.get("core_temperature_c", 0.0)
        rad_temp = thermal_data.get("radiator_temperature_c", 0.0)
        thermal_status = thermal_data.get("status_message", "UNKNOWN")
        
        integrity = frame_data.get("integrity_percentage", 100.0)
        stress = frame_data.get("stress_level", 0.0)
        vibration = frame_data.get("vibration_level", 0.0)
        
        # Рисуем рамку и заголовок
        runtime = time.time() - self.start_time
        self._add_to_buffer("=" * self.width)
        self._add_to_buffer(f"QIKI Robot Simulation (Runtime: {runtime:.1f}s, FPS: {fps:.1f})")
        self._add_to_buffer("=" * self.width)
        
        # Навигационная информация с улучшенным форматированием
        self._add_to_buffer("\n=== Navigation ===")
        # Позиция с обозначением осей
        self._add_to_buffer(f"Position:    X:{pos[0]:8.2f} Y:{pos[1]:8.2f} Z:{pos[2]:8.2f} m")
        
        # Скорость с величиной и направлением
        vel_mag = np.linalg.norm(vel)
        self._add_to_buffer(f"Velocity:    X:{vel[0]:8.2f} Y:{vel[1]:8.2f} Z:{vel[2]:8.2f} m/s")
        self._add_to_buffer(f"Speed:       {self._format_si(vel_mag, 'm/s')} ({'^' if vel_mag > 10 else '-' if vel_mag > 0.1 else '.'} )")
        
        # Ускорение с индикатором перегрузки
        acc_mag = np.linalg.norm(acc)
        g_force = acc_mag / 9.81
        self._add_to_buffer(f"Acceleration:{self._format_si(acc_mag, 'm/s²')} ({g_force:4.1f}G)")
        
        # Угловая скорость с индикатором вращения
        ang_vel_mag = np.linalg.norm(ang_vel)
        rot_indicator = '⭮' if ang_vel_mag > 0.1 else '•'
        self._add_to_buffer(f"Angular Vel: {self._format_si(ang_vel_mag, 'rad/s')} {rot_indicator}")
        
        # Системная информация с улучшенным форматированием и индикаторами состояния
        self._add_to_buffer("\n=== Systems Status ===")
        
        # Тепловой режим
        temp_status = "[ OK ]" if temp < 60 else "[WARN]" if temp < 80 else "[CRIT]"
        self._add_to_buffer(f"Thermal:     {temp_status} Core:{temp:6.1f}°C  Radiator:{rad_temp:6.1f}°C")
        self._add_to_buffer(f"            Status: {thermal_status}")
        
        # Энергосистема
        power_efficiency = solar_power / (power_draw if power_draw > 0 else 1)
        power_status = "[ OK ]" if power_efficiency >= 1 else "[WARN]" if power_efficiency >= 0.5 else "[CRIT]"
        self._add_to_buffer(f"Power:       {power_status} Draw:{self._format_si(power_draw, 'W')}  Solar:{self._format_si(solar_power, 'W')}")
        self._add_to_buffer(f"            Efficiency: {power_efficiency:6.1%}")
        
        # Состояние конструкции
        frame_status = "[ OK ]" if integrity > 90 else "[WARN]" if integrity > 50 else "[CRIT]"
        self._add_to_buffer(f"Frame:       {frame_status} Integrity:{integrity:6.1f}%  Stress:{stress:6.2f}")
        self._add_to_buffer(f"            Vibration: {vibration:6.2f} Hz")
        
        # Сейчас мы не выводим карту, так как она не информативна
        self._add_to_buffer("\n=== Extended Navigation ===")
        self._add_to_buffer(f"Absolute Position (m):")
        self._add_to_buffer(f"  X: {pos[0]:8.2f}  Y: {pos[1]:8.2f}  Z: {pos[2]:8.2f}")
        self._add_to_buffer(f"Absolute Velocity (m/s):")
        self._add_to_buffer(f"  X: {vel[0]:8.2f}  Y: {vel[1]:8.2f}  Z: {vel[2]:8.2f}")
        self._add_to_buffer(f"Acceleration (m/s²):")
        self._add_to_buffer(f"  X: {acc[0]:8.2f}  Y: {acc[1]:8.2f}  Z: {acc[2]:8.2f}")
        
        # Индикаторы состояния
        bar_width = 30  # Длина полос прогресса
        
        self._add_to_buffer("\n=== Status Bars ===")
        
        # Батарея
        bat_bar = int(bat * bar_width / 100)
        self._add_to_buffer(f"Battery:   [{('=' * bat_bar) + (' ' * (bar_width-bat_bar))}] {bat:3.0f}%")
        
        # Резервная батарея
        bat_res_bar = int(bat_reserve * bar_width / 100)
        self._add_to_buffer(f"Reserve:   [{('=' * bat_res_bar) + (' ' * (bar_width-bat_res_bar))}] {bat_reserve:3.0f}%")
        
        # Температура
        temp_norm = min(100, max(0, temp)) / 100  # Нормализуем до 0-100%
        temp_bar = int(temp_norm * bar_width)
        self._add_to_buffer(f"Core Temp: [{('=' * temp_bar) + (' ' * (bar_width-temp_bar))}] {temp:3.0f}°C")
        
        # Целостность конструкции
        integrity_bar = int(integrity * bar_width / 100)
        self._add_to_buffer(f"Integrity: [{('=' * integrity_bar) + (' ' * (bar_width-integrity_bar))}] {integrity:3.0f}%")
        
        # Информация об окружающей среде
        try:
            solar_irradiance = environment.get_solar_irradiance() if environment else 0.0
            gravity = environment.gravity if environment else np.array([0.0, 0.0, -9.81])
            drag_coeff = environment.air_drag_coefficient if environment else 0.0
        except AttributeError:
            solar_irradiance = 0.0
            gravity = np.array([0.0, 0.0, -9.81])
            drag_coeff = 0.0
        
        self._add_to_buffer("\n=== Environment ===")
        self._add_to_buffer(f"Solar:       {self._format_si(solar_irradiance, 'W/m²')}")
        self._add_to_buffer(f"Gravity:     {self._format_si(np.linalg.norm(gravity), 'm/s²')}")
        if drag_coeff > 0:
            self._add_to_buffer(f"Drag Coeff:  {drag_coeff:.2e}")
        
        # Вывод буфера
        self._flush_buffer()
        
        # Обновляем время последнего кадра
        self.last_frame_time = time.time()
        
    def _format_si(self, value: float, unit: str = "", precision: int = 1) -> str:
        """Форматирует число с СИ префиксом"""
        prefixes = ['', 'k', 'M', 'G']
        prefix_idx = 0
        abs_value = abs(value)
        
        while abs_value >= 1000.0 and prefix_idx < len(prefixes) - 1:
            abs_value /= 1000.0
            prefix_idx += 1
            
        return f"{value/pow(1000, prefix_idx):{precision+4}.{precision}f} {prefixes[prefix_idx]}{unit}"
    
    def _format_si(self, value: float, unit: str = "", precision: int = 1) -> str:
        """Форматирует число с СИ префиксом"""
        prefixes = ['', 'k', 'M', 'G']
        prefix_idx = 0
        abs_value = abs(value)
        
        while abs_value >= 1000.0 and prefix_idx < len(prefixes) - 1:
            abs_value /= 1000.0
            prefix_idx += 1
            
        return f"{value/pow(1000, prefix_idx):{precision+4}.{precision}f} {prefixes[prefix_idx]}{unit}"
