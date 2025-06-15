"""
QIKI Display Manager - Эталонный интерфейс визуализации
=====================================================

Унифицированная система отображения состояния симуляции с поддержкой:
- Интерактивного управления в реальном времени
- Множественных режимов отображения
- Оптимизированного рендеринга (delta updates)
- Интеграции с операционной системой
- Расширяемой архитектуры

Автор: QIKI Project Team
Версия: 2.0
"""

import sys
import os
import time
import asyncio
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Callable
from collections import deque
import numpy as np

from logger import Logger


class DisplayMode(Enum):
    """Режимы отображения информации."""
    MINIMAL = auto()      # Минимальная информация
    STANDARD = auto()     # Стандартный набор данных
    DETAILED = auto()     # Подробная информация
    DEBUG = auto()        # Отладочная информация
    TELEMETRY = auto()    # Фокус на телеметрии


class InputCommand(Enum):
    """Доступные команды ввода."""
    PAUSE = auto()
    RESUME = auto()
    QUIT = auto()
    MODE_CYCLE = auto()
    SPEED_UP = auto()
    SPEED_DOWN = auto()
    RESET_VIEW = auto()
    SAVE_DATA = auto()
    EMERGENCY_STOP = auto()


@dataclass
class DisplayFrame:
    """Структура данных для одного кадра отображения."""
    timestamp: float
    bot_position: np.ndarray
    bot_velocity: np.ndarray
    bot_acceleration: np.ndarray
    bot_orientation: np.ndarray
    target_position: np.ndarray
    system_status: Dict[str, Any]
    sensor_data: Dict[str, Any]
    simulation_state: Dict[str, Any]
    control_status: Optional[Dict[str, Any]] = None  # Добавляем статус управления


@dataclass
class DisplayConfig:
    """Конфигурация системы отображения."""
    # Размеры и обновление
    width: int = 120
    height: int = 30
    update_rate: float = 10.0  # FPS
    
    # Режимы
    default_mode: DisplayMode = DisplayMode.STANDARD
    auto_scroll: bool = True
    show_fps: bool = True
    
    # Цвета и стили (ANSI коды)
    colors_enabled: bool = True
    color_scheme: Dict[str, str] = field(default_factory=lambda: {
        'header': '\033[1;36m',    # Cyan bold
        'warning': '\033[1;33m',   # Yellow bold
        'error': '\033[1;31m',     # Red bold
        'success': '\033[1;32m',   # Green bold
        'data': '\033[0;37m',      # White
        'reset': '\033[0m'         # Reset
    })
    
    # Пороговые значения для предупреждений
    warning_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'battery_low': 20.0,
        'temperature_high': 85.0,
        'integrity_low': 80.0
    })


class DisplayRenderer(ABC):
    """Абстрактный базовый класс для рендереров."""
    
    @abstractmethod
    def render(self, frame: DisplayFrame, config: DisplayConfig) -> List[str]:
        """Рендерит кадр в список строк для вывода."""
        pass
    
    @abstractmethod
    def get_required_height(self) -> int:
        """Возвращает требуемую высоту для рендерера."""
        pass


class StandardRenderer(DisplayRenderer):
    """Стандартный рендерер с базовой информацией."""
    
    def __init__(self):
        self.logger = Logger.get_logger("standard_renderer")
    
    def render(self, frame: DisplayFrame, config: DisplayConfig) -> List[str]:
        lines = []
        colors = config.color_scheme if config.colors_enabled else {k: '' for k in config.color_scheme}
        
        # Заголовок
        lines.append(f"{colors['header']}{'=' * config.width}{colors['reset']}")
        lines.append(f"{colors['header']}QIKI SIMULATION - T={frame.timestamp:.1f}s{colors['reset']}")
        lines.append(f"{colors['header']}{'=' * config.width}{colors['reset']}")
        
        # Основная информация о роботе
        pos = frame.bot_position
        vel = frame.bot_velocity
        target = frame.target_position
        distance = np.linalg.norm(target - pos)
        
        lines.append(f"{colors['data']}Robot Position  : [{pos[0]:8.2f}, {pos[1]:8.2f}, {pos[2]:8.2f}] m{colors['reset']}")
        lines.append(f"{colors['data']}Target Position : [{target[0]:8.2f}, {target[1]:8.2f}, {target[2]:8.2f}] m{colors['reset']}")
        lines.append(f"{colors['data']}Distance to Target: {distance:8.2f} m{colors['reset']}")
        lines.append(f"{colors['data']}Velocity        : [{vel[0]:8.2f}, {vel[1]:8.2f}, {vel[2]:8.2f}] m/s{colors['reset']}")
        lines.append(f"{colors['data']}Speed           : {np.linalg.norm(vel):8.2f} m/s{colors['reset']}")
        
        lines.append("")
        
        # Статус систем
        power = frame.system_status.get('power', {})
        thermal = frame.system_status.get('thermal', {})
        frame_sys = frame.system_status.get('frame', {})
        
        battery = power.get('total_battery_percentage', 0)
        temp = thermal.get('core_temperature_c', 0)
        integrity = frame_sys.get('structural_integrity', 100)
        
        # Цветное отображение статуса
        battery_color = colors['error'] if battery < config.warning_thresholds['battery_low'] else colors['success']
        temp_color = colors['error'] if temp > config.warning_thresholds['temperature_high'] else colors['data']
        integrity_color = colors['error'] if integrity < config.warning_thresholds['integrity_low'] else colors['success']
        
        lines.append(f"{colors['header']}SYSTEM STATUS:{colors['reset']}")
        lines.append(f"  {battery_color}Battery      : {battery:6.1f}% {power.get('status', 'N/A')}{colors['reset']}")
        lines.append(f"  {temp_color}Temperature  : {temp:6.1f}°C {thermal.get('status_message', 'N/A')}{colors['reset']}")
        lines.append(f"  {integrity_color}Integrity    : {integrity:6.1f}%{colors['reset']}")
        
        # Статус управления (если доступен)
        if frame.control_status and frame.control_status.get('status') == 'active':
            control = frame.control_status
            lines.append("")
            lines.append(f"{colors['header']}CONTROL STATUS:{colors['reset']}")
            
            mode_color = colors['warning'] if control.get('manual_override') else colors['data']
            lines.append(f"  {mode_color}Control Mode : {control.get('mode', 'N/A')}{colors['reset']}")
            
            if control.get('emergency_stop'):
                lines.append(f"  {colors['error']}EMERGENCY STOP ACTIVE{colors['reset']}")
            
            active_interfaces = control.get('active_interfaces', 0)
            lines.append(f"  {colors['data']}Input Methods: {active_interfaces} active{colors['reset']}")
        
        return lines
    
    def get_required_height(self) -> int:
        return 15


class DetailedRenderer(DisplayRenderer):
    """Подробный рендерер с расширенной информацией."""
    
    def __init__(self):
        self.logger = Logger.get_logger("detailed_renderer")
    
    def render(self, frame: DisplayFrame, config: DisplayConfig) -> List[str]:
        lines = []
        colors = config.color_scheme if config.colors_enabled else {k: '' for k in config.color_scheme}
        
        # Используем стандартный рендерер как базу
        standard = StandardRenderer()
        lines.extend(standard.render(frame, config))
        
        # Добавляем подробную информацию
        lines.append("")
        lines.append(f"{colors['header']}DETAILED TELEMETRY:{colors['reset']}")
        
        # Ускорение
        acc = frame.bot_acceleration
        lines.append(f"  {colors['data']}Acceleration : [{acc[0]:8.2f}, {acc[1]:8.2f}, {acc[2]:8.2f}] m/s²{colors['reset']}")
        
        # Ориентация
        quat = frame.bot_orientation
        lines.append(f"  {colors['data']}Orientation  : [{quat[0]:6.3f}, {quat[1]:6.3f}, {quat[2]:6.3f}, {quat[3]:6.3f}]{colors['reset']}")
        
        # Сенсоры
        if 'power' in frame.sensor_data:
            solar = frame.sensor_data['power'].get('solar_irradiance_w_m2', 0)
            lines.append(f"  {colors['data']}Solar Power  : {solar:8.1f} W/m²{colors['reset']}")
        
        if 'thermal' in frame.sensor_data:
            radiator_temp = frame.sensor_data['thermal'].get('radiator_temperature_c', 0)
            lines.append(f"  {colors['data']}Radiator Temp: {radiator_temp:8.1f}°C{colors['reset']}")
        
        if 'frame' in frame.sensor_data:
            stress = frame.sensor_data['frame'].get('stress_level', 0)
            lines.append(f"  {colors['data']}Stress Level : {stress:8.3f}{colors['reset']}")
        
        return lines
    
    def get_required_height(self) -> int:
        return 25


class TelemetryRenderer(DisplayRenderer):
    """Рендерер с фокусом на телеметрии и графиках."""
    
    def __init__(self):
        self.logger = Logger.get_logger("telemetry_renderer")
        self.history: deque = deque(maxlen=60)  # 60 последних значений
    
    def render(self, frame: DisplayFrame, config: DisplayConfig) -> List[str]:
        lines = []
        colors = config.color_scheme if config.colors_enabled else {k: '' for k in config.color_scheme}
        
        # Добавляем текущий кадр в историю
        self.history.append(frame)
        
        lines.append(f"{colors['header']}TELEMETRY DASHBOARD - T={frame.timestamp:.1f}s{colors['reset']}")
        lines.append(f"{colors['header']}{'=' * config.width}{colors['reset']}")
        
        # Мини-графики ASCII
        if len(self.history) > 10:
            lines.extend(self._render_ascii_graphs(colors))
        
        # Текущие значения
        lines.append("")
        lines.append(f"{colors['header']}CURRENT VALUES:{colors['reset']}")
        
        power = frame.system_status.get('power', {})
        thermal = frame.system_status.get('thermal', {})
        
        battery = power.get('total_battery_percentage', 0)
        temp = thermal.get('core_temperature_c', 0)
        speed = np.linalg.norm(frame.bot_velocity)
        
        lines.append(f"  Battery: {battery:6.1f}%  Temperature: {temp:6.1f}°C  Speed: {speed:6.2f} m/s")
        
        return lines
    
    def _render_ascii_graphs(self, colors: Dict[str, str]) -> List[str]:
        """Рендерит простые ASCII графики."""
        lines = []
        
        # График батареи
        battery_values = [
            frame.system_status.get('power', {}).get('total_battery_percentage', 0) 
            for frame in list(self.history)[-30:]
        ]
        
        if battery_values:
            lines.append(f"{colors['data']}Battery Level (last 30 samples):{colors['reset']}")
            lines.append(self._create_sparkline(battery_values, 0, 100))
            lines.append("")
        
        return lines
    
    def _create_sparkline(self, values: List[float], min_val: float, max_val: float) -> str:
        """Создает простой ASCII спарклайн."""
        if not values:
            return ""
        
        chars = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        
        # Нормализуем значения
        normalized = []
        for val in values:
            if max_val > min_val:
                norm = (val - min_val) / (max_val - min_val)
            else:
                norm = 0.5
            norm = max(0, min(1, norm))  # Ограничиваем 0-1
            normalized.append(norm)
        
        # Преобразуем в символы
        sparkline = ""
        for norm in normalized:
            char_idx = int(norm * (len(chars) - 1))
            sparkline += chars[char_idx]
        
        return f"  {sparkline} ({min(values):.1f} - {max(values):.1f})"
    
    def get_required_height(self) -> int:
        return 20


class InputHandler:
    """Обработчик пользовательского ввода."""
    
    def __init__(self):
        self.logger = Logger.get_logger("input_handler")
        self.command_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        
        # Мапинг клавиш к командам
        self.key_mapping = {
            'p': InputCommand.PAUSE,
            'r': InputCommand.RESUME,
            'q': InputCommand.QUIT,
            'm': InputCommand.MODE_CYCLE,
            '+': InputCommand.SPEED_UP,
            '-': InputCommand.SPEED_DOWN,
            '0': InputCommand.RESET_VIEW,
            's': InputCommand.SAVE_DATA,
            'e': InputCommand.EMERGENCY_STOP,
        }
    
    async def start(self):
        """Запускает обработчик ввода в отдельном потоке."""
        self.running = True
        self.input_thread = threading.Thread(target=self._input_worker, daemon=True)
        self.input_thread.start()
        self.logger.info("Input handler started")
    
    def _input_worker(self):
        """Рабочий поток для чтения ввода."""
        import select
        
        while self.running:
            try:
                # Неблокирующее чтение для Unix систем
                if os.name != 'nt' and select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    if key in self.key_mapping:
                        asyncio.run_coroutine_threadsafe(
                            self.command_queue.put(self.key_mapping[key]),
                            asyncio.get_event_loop()
                        )
                elif os.name == 'nt':  # Windows
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8').lower()
                        if key in self.key_mapping:
                            asyncio.run_coroutine_threadsafe(
                                self.command_queue.put(self.key_mapping[key]),
                                asyncio.get_event_loop()
                            )
                
                time.sleep(0.05)  # Небольшая пауза
            except Exception as e:
                self.logger.error(f"Input handler error: {e}")
    
    async def get_command(self) -> Optional[InputCommand]:
        """Получает команду из очереди (неблокирующе)."""
        try:
            return await asyncio.wait_for(self.command_queue.get(), timeout=0.001)
        except asyncio.TimeoutError:
            return None
    
    def stop(self):
        """Останавливает обработчик ввода."""
        self.running = False


class DisplayManager:
    """
    Главный менеджер системы отображения.
    Координирует рендереры, ввод и обновления экрана.
    """
    
    def __init__(self, config: DisplayConfig = None):
        self.logger = Logger.get_logger("display_manager")
        self.config = config or DisplayConfig()
        
        # Рендереры для разных режимов
        self.renderers = {
            DisplayMode.MINIMAL: StandardRenderer(),
            DisplayMode.STANDARD: StandardRenderer(),
            DisplayMode.DETAILED: DetailedRenderer(),
            DisplayMode.TELEMETRY: TelemetryRenderer(),
            DisplayMode.DEBUG: DetailedRenderer(),  # Пока используем detailed
        }
        
        self.current_mode = self.config.default_mode
        self.input_handler = InputHandler()
        
        # Статистика производительности
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        
        # Кеширование для оптимизации
        self.last_frame_lines: List[str] = []
        self.screen_buffer: List[str] = []
        
        # Колбэки для команд
        self.command_callbacks: Dict[InputCommand, Callable] = {}
        
        self.logger.info(f"DisplayManager initialized in {self.current_mode.name} mode")
    
    def register_command_callback(self, command: InputCommand, callback: Callable):
        """Регистрирует колбэк для команды."""
        self.command_callbacks[command] = callback
    
    async def start(self):
        """Запускает менеджер отображения."""
        await self.input_handler.start()
        self._setup_console()
        self.logger.info("Display manager started")
    
    def _setup_console(self):
        """Настраивает консоль для оптимального отображения."""
        if os.name != 'nt':  # Unix
            os.system('clear')
            # Скрываем курсор
            sys.stdout.write('\033[?25l')
            sys.stdout.flush()
        else:  # Windows
            os.system('cls')
    
    async def update(self, frame: DisplayFrame):
        """Обновляет отображение новым кадром."""
        # Обрабатываем команды ввода
        await self._process_input()
        
        # Рендерим кадр
        renderer = self.renderers[self.current_mode]
        new_lines = renderer.render(frame, self.config)
        
        # Добавляем информацию о режиме и командах
        new_lines.extend(self._render_footer())
        
        # Оптимизированное обновление экрана (delta update)
        self._update_screen(new_lines)
        
        # Обновляем статистику
        self._update_fps()
    
    async def _process_input(self):
        """Обрабатывает команды пользователя."""
        command = await self.input_handler.get_command()
        if command:
            self.logger.debug(f"Received command: {command.name}")
            
            # Встроенные команды
            if command == InputCommand.MODE_CYCLE:
                self._cycle_mode()
            elif command == InputCommand.QUIT:
                # Должен обрабатываться извне
                pass
            
            # Пользовательские колбэки
            if command in self.command_callbacks:
                try:
                    await self.command_callbacks[command]()
                except Exception as e:
                    self.logger.error(f"Error executing command callback: {e}")
    
    def _cycle_mode(self):
        """Переключает режим отображения."""
        modes = list(DisplayMode)
        current_idx = modes.index(self.current_mode)
        next_idx = (current_idx + 1) % len(modes)
        self.current_mode = modes[next_idx]
        self.logger.info(f"Switched to {self.current_mode.name} mode")
    
    def _render_footer(self) -> List[str]:
        """Рендерит нижнюю часть экрана с информацией о командах."""
        lines = []
        colors = self.config.color_scheme if self.config.colors_enabled else {k: '' for k in self.config.color_scheme}
        
        lines.append("")
        lines.append(f"{colors['header']}{'─' * self.config.width}{colors['reset']}")
        
        # Информация о режиме и FPS
        mode_info = f"Mode: {self.current_mode.name}"
        fps_info = f"FPS: {self.fps:.1f}" if self.config.show_fps else ""
        status_line = f"{mode_info:30} {fps_info:>20}"
        lines.append(f"{colors['data']}{status_line}{colors['reset']}")
        
        # Команды
        commands = "Commands: [P]ause [R]esume [M]ode [Q]uit [+/-]Speed [S]ave [E]mergency"
        lines.append(f"{colors['data']}{commands}{colors['reset']}")
        
        return lines
    
    def _update_screen(self, new_lines: List[str]):
        """Обновляет экран с оптимизацией (только измененные строки)."""
        if os.name != 'nt':  # Unix - используем ANSI коды для позиционирования
            # Перемещаем курсор в начало
            sys.stdout.write('\033[H')
            
            # Обновляем только измененные строки
            for i, line in enumerate(new_lines):
                if i >= len(self.last_frame_lines) or line != self.last_frame_lines[i]:
                    sys.stdout.write(f'\033[{i+1};1H')  # Позиционируем курсор
                    sys.stdout.write(line)
                    sys.stdout.write('\033[K')  # Очищаем до конца строки
        
        else:  # Windows - полная перерисовка
            os.system('cls')
            for line in new_lines:
                print(line)
        
        sys.stdout.flush()
        self.last_frame_lines = new_lines.copy()
    
    def _update_fps(self):
        """Обновляет счетчик FPS."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def stop(self):
        """Останавливает менеджер отображения."""
        self.input_handler.stop()
        
        # Восстанавливаем консоль
        if os.name != 'nt':
            # Показываем курсор
            sys.stdout.write('\033[?25h')
            sys.stdout.flush()
        
        self.logger.info("Display manager stopped")
    
    def get_current_mode(self) -> DisplayMode:
        """Возвращает текущий режим отображения."""
        return self.current_mode
    
    def set_mode(self, mode: DisplayMode):
        """Устанавливает режим отображения."""
        if mode in self.renderers:
            self.current_mode = mode
            self.logger.info(f"Display mode set to {mode.name}")
        else:
            self.logger.warning(f"Unknown display mode: {mode}")
