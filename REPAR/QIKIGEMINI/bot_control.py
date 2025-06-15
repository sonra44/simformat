"""
QIKI Bot Control System - Универсальное управление ботом
======================================================

Система прямого управления ботом с поддержкой:
- Кроссплатформенного ввода (Android, Windows, Linux)
- Режимов управления (ручной/автоматический/смешанный)
- Безопасного управления без root-прав
- Команд навигации и маневрирования
- Виртуального джойстика для мобильных устройств

Автор: QIKI Project Team
Версия: 1.0
"""

import asyncio
import sys
import os
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional
import numpy as np

from logger import Logger
from config import config


class ControlMode(Enum):
    """Режимы управления ботом."""
    AUTONOMOUS = auto()    # Полностью автономный
    MANUAL = auto()        # Ручное управление
    ASSISTED = auto()      # Ассистированное управление
    MIXED = auto()         # Смешанный режим


class CommandType(Enum):
    """Типы команд управления."""
    # Движение
    MOVE_FORWARD = auto()
    MOVE_BACKWARD = auto()
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    MOVE_UP = auto()
    MOVE_DOWN = auto()
    
    # Вращение
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    PITCH_UP = auto()
    PITCH_DOWN = auto()
    ROLL_LEFT = auto()
    ROLL_RIGHT = auto()
    
    # Навигация
    GOTO_POSITION = auto()
    FOLLOW_TARGET = auto()
    ORBIT_TARGET = auto()
    HOVER = auto()
    LAND = auto()
    
    # Системные
    EMERGENCY_STOP = auto()
    RESET_POSITION = auto()
    CHANGE_MODE = auto()
    CALIBRATE = auto()


@dataclass
class ControlCommand:
    """Структура команды управления."""
    command_type: CommandType
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # 0 = высший приоритет
    duration: Optional[float] = None  # Продолжительность выполнения
    source: str = "unknown"  # Источник команды


@dataclass
class ControlState:
    """Состояние системы управления."""
    mode: ControlMode = ControlMode.AUTONOMOUS
    is_active: bool = True
    last_command_time: float = 0.0
    manual_override: bool = False
    emergency_stop: bool = False
    
    # Текущие команды
    thrust_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Целевые значения
    target_position: Optional[np.ndarray] = None
    target_velocity: Optional[np.ndarray] = None
    target_attitude: Optional[np.ndarray] = None


class InputInterface(ABC):
    """Абстрактный интерфейс для ввода команд."""
    
    @abstractmethod
    async def start(self) -> None:
        """Запускает интерфейс ввода."""
        pass
    
    @abstractmethod
    async def get_command(self) -> Optional[ControlCommand]:
        """Получает команду от пользователя."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Останавливает интерфейс ввода."""
        pass


class KeyboardInterface(InputInterface):
    """Интерфейс управления с клавиатуры (для Desktop)."""
    
    def __init__(self):
        self.logger = Logger.get_logger("keyboard_interface")
        self.command_queue = asyncio.Queue()
        self.running = False
        self.input_thread = None
        
        # Маппинг клавиш
        self.key_mapping = {
            # WASD для движения
            'w': (CommandType.MOVE_FORWARD, {}),
            's': (CommandType.MOVE_BACKWARD, {}),
            'a': (CommandType.MOVE_LEFT, {}),
            'd': (CommandType.MOVE_RIGHT, {}),
            
            # Вертикальное движение
            'q': (CommandType.MOVE_UP, {}),
            'e': (CommandType.MOVE_DOWN, {}),
            
            # Вращение (стрелки)
            'j': (CommandType.TURN_LEFT, {}),
            'l': (CommandType.TURN_RIGHT, {}),
            'i': (CommandType.PITCH_UP, {}),
            'k': (CommandType.PITCH_DOWN, {}),
            'u': (CommandType.ROLL_LEFT, {}),
            'o': (CommandType.ROLL_RIGHT, {}),
            
            # Специальные команды
            'h': (CommandType.HOVER, {}),
            'r': (CommandType.RESET_POSITION, {}),
            'x': (CommandType.EMERGENCY_STOP, {}),
            'm': (CommandType.CHANGE_MODE, {}),
            ' ': (CommandType.HOVER, {}),  # Пробел для зависания
        }
    
    async def start(self) -> None:
        """Запускает клавиатурный интерфейс."""
        self.running = True
        self.input_thread = threading.Thread(target=self._input_worker, daemon=True)
        self.input_thread.start()
        self.logger.info("Keyboard interface started")
        
        # Инструкции для пользователя
        print("\\n🎮 QIKI Bot Control - Keyboard Interface")
        print("==========================================")
        print("Movement:     W/S/A/D - Forward/Back/Left/Right")
        print("Vertical:     Q/E - Up/Down")
        print("Rotation:     I/K/J/L - Pitch/Turn")
        print("              U/O - Roll Left/Right")
        print("Commands:     H - Hover, R - Reset, X - Emergency Stop")
        print("              M - Change Mode, SPACE - Hover")
        print("==========================================\\n")
    
    def _input_worker(self):
        """Рабочий поток для чтения клавиатуры."""
        try:
            if os.name != 'nt':  # Unix/Linux/Android
                self._unix_input_loop()
            else:  # Windows
                self._windows_input_loop()
        except Exception as e:
            self.logger.error(f"Input worker error: {e}")
    
    def _unix_input_loop(self):
        """Цикл ввода для Unix-подобных систем."""
        import select
        import termios
        import tty
        
        try:
            # Настраиваем терминал для посимвольного ввода
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            
            while self.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    self._process_key(key)
                    
        except Exception as e:
            self.logger.error(f"Unix input error: {e}")
        finally:
            # Восстанавливаем настройки терминала
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except Exception:
                pass
    
    def _windows_input_loop(self):
        """Цикл ввода для Windows."""
        try:
            import msvcrt
            
            while self.running:
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                    self._process_key(key)
                else:
                    time.sleep(0.05)
                    
        except Exception as e:
            self.logger.error(f"Windows input error: {e}")
    
    def _process_key(self, key: str):
        """Обрабатывает нажатие клавиши."""
        if key in self.key_mapping:
            command_type, params = self.key_mapping[key]
            command = ControlCommand(
                command_type=command_type,
                parameters=params,
                source="keyboard"
            )
            
            # Неблокирующе добавляем команду в очередь
            try:
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(
                    self.command_queue.put(command), loop
                )
            except Exception as e:
                self.logger.error(f"Error queuing command: {e}")
    
    async def get_command(self) -> Optional[ControlCommand]:
        """Получает команду из очереди."""
        try:
            return await asyncio.wait_for(self.command_queue.get(), timeout=0.001)
        except asyncio.TimeoutError:
            return None
    
    def stop(self) -> None:
        """Останавливает интерфейс."""
        self.running = False


class TouchInterface(InputInterface):
    """Интерфейс для сенсорных устройств (Android, iOS)."""
    
    def __init__(self):
        self.logger = Logger.get_logger("touch_interface")
        self.command_queue = asyncio.Queue()
        self.running = False
        
        # Виртуальный джойстик с параметрами из конфигурации
        self.joystick_center = (0.0, 0.0)
        self.joystick_deadzone = config.TOUCH_DEADZONE_RADIUS
        self.joystick_sensitivity = config.JOYSTICK_SENSITIVITY
        
        # Android-специфичные настройки
        self.android_mode = config.ANDROID_NO_ROOT_MODE
        self.safe_input_mode = config.ANDROID_SAFE_INPUT_MODE
        
        # Поддержка различных типов сенсорного ввода
        self.touch_modes = {
            'joystick': self._joystick_mode,
            'directional': self._directional_mode,
            'gesture': self._gesture_mode
        }
        self.current_touch_mode = 'joystick' if not self.android_mode else 'directional'
    
    async def start(self) -> None:
        """Запускает сенсорный интерфейс."""
        self.running = True
        self.logger.info(f"Touch interface started in {self.current_touch_mode} mode")
        
        if self.android_mode:
            print("\\n📱 QIKI Bot Control - Android Touch Interface (No Root)")
            print("======================================================")
            print("🚫 Root-free mode: Using safe input methods")
            print("📱 Optimized for mobile devices")
            print(f"🎮 Mode: {self.current_touch_mode}")
        else:
            print("\\n📱 QIKI Bot Control - Touch Interface")
            print("=====================================")
        
        print("Virtual Joystick: Use coordinate input")
        print("Commands: Send via text interface")
        print("Touch modes: joystick, directional, gesture")
        print("=====================================\\n")
    
    def _joystick_mode(self, x: float, y: float, action: str = "move"):
        """Режим виртуального джойстика (требует точных координат)."""
        dx = x - self.joystick_center[0]
        dy = y - self.joystick_center[1]
        
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance > self.joystick_deadzone:
            dx_norm = (dx / distance) * min(distance, 1.0) * self.joystick_sensitivity
            dy_norm = (dy / distance) * min(distance, 1.0) * self.joystick_sensitivity
            
            return ControlCommand(
                command_type=CommandType.MOVE_FORWARD,
                parameters={
                    'velocity_x': dx_norm,
                    'velocity_y': dy_norm,
                    'relative': True
                },
                source="touch_joystick"
            )
        return None
    
    def _directional_mode(self, x: float, y: float, action: str = "move"):
        """Упрощенный режим направления (безопасен для Android)."""
        # Упрощенное управление: 4 направления
        if abs(x) > abs(y):
            # Горизонтальное движение
            if x > 0.5:
                return ControlCommand(
                    command_type=CommandType.MOVE_RIGHT,
                    parameters={},
                    source="touch_directional"
                )
            elif x < -0.5:
                return ControlCommand(
                    command_type=CommandType.MOVE_LEFT,
                    parameters={},
                    source="touch_directional"
                )
        else:
            # Вертикальное движение
            if y > 0.5:
                return ControlCommand(
                    command_type=CommandType.MOVE_FORWARD,
                    parameters={},
                    source="touch_directional"
                )
            elif y < -0.5:
                return ControlCommand(
                    command_type=CommandType.MOVE_BACKWARD,
                    parameters={},
                    source="touch_directional"
                )
        return None
    
    def _gesture_mode(self, x: float, y: float, action: str = "move"):
        """Режим жестов (для опытных пользователей)."""
        # Комплексные жесты для расширенного управления
        if action == "swipe_up":
            return ControlCommand(CommandType.MOVE_UP, {}, "touch_gesture")
        elif action == "swipe_down":
            return ControlCommand(CommandType.MOVE_DOWN, {}, "touch_gesture")
        elif action == "tap":
            return ControlCommand(CommandType.HOVER, {}, "touch_gesture")
        elif action == "double_tap":
            return ControlCommand(CommandType.EMERGENCY_STOP, {}, "touch_gesture")
        
        # Fallback к джойстику
        return self._joystick_mode(x, y, action)
    
    async def get_command(self) -> Optional[ControlCommand]:
        """Получает команду от сенсорного интерфейса."""
        try:
            return await asyncio.wait_for(self.command_queue.get(), timeout=0.001)
        except asyncio.TimeoutError:
            return None
    
    def process_touch(self, x: float, y: float, action: str = "move"):
        """Обрабатывает сенсорный ввод с учетом текущего режима."""
        if not self.running:
            return
        
        # Выбираем обработчик в зависимости от режима
        handler = self.touch_modes.get(self.current_touch_mode, self._joystick_mode)
        command = handler(x, y, action)
        
        if command:
            # Добавляем в очередь
            try:
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(
                    self.command_queue.put(command), loop
                )
                
                if self.android_mode:
                    self.logger.debug(f"Android touch: {action} at ({x:.2f}, {y:.2f}) -> {command.command_type.name}")
                
            except Exception as e:
                self.logger.error(f"Error queuing touch command: {e}")
    
    def set_touch_mode(self, mode: str):
        """Меняет режим сенсорного управления."""
        if mode in self.touch_modes:
            self.current_touch_mode = mode
            self.logger.info(f"Touch mode changed to: {mode}")
            return True
        return False
    
    def stop(self) -> None:
        """Останавливает интерфейс."""
        self.running = False


class TextInterface(InputInterface):
    """Текстовый интерфейс команд (универсальный для всех платформ)."""
    
    def __init__(self):
        self.logger = Logger.get_logger("text_interface")
        self.command_queue = asyncio.Queue()
        self.running = False
        self.input_thread = None
        
        # Текстовые команды
        self.text_commands = {
            # Движение
            'forward': (CommandType.MOVE_FORWARD, {}),
            'backward': (CommandType.MOVE_BACKWARD, {}),
            'left': (CommandType.MOVE_LEFT, {}),
            'right': (CommandType.MOVE_RIGHT, {}),
            'up': (CommandType.MOVE_UP, {}),
            'down': (CommandType.MOVE_DOWN, {}),
            
            # Навигация
            'hover': (CommandType.HOVER, {}),
            'land': (CommandType.LAND, {}),
            'reset': (CommandType.RESET_POSITION, {}),
            'stop': (CommandType.EMERGENCY_STOP, {}),
            'emergency': (CommandType.EMERGENCY_STOP, {}),
            
            # Режимы
            'auto': (CommandType.CHANGE_MODE, {'mode': ControlMode.AUTONOMOUS}),
            'manual': (CommandType.CHANGE_MODE, {'mode': ControlMode.MANUAL}),
            'assisted': (CommandType.CHANGE_MODE, {'mode': ControlMode.ASSISTED}),
            
            # Координаты
            'goto': (CommandType.GOTO_POSITION, {}),  # Требует параметры
        }
    
    async def start(self) -> None:
        """Запускает текстовый интерфейс."""
        self.running = True
        self.input_thread = threading.Thread(target=self._input_worker, daemon=True)
        self.input_thread.start()
        self.logger.info("Text interface started")
        
        print("\\n💬 QIKI Bot Control - Text Interface")
        print("====================================")
        print("Commands: forward, backward, left, right, up, down")
        print("          hover, land, reset, stop, emergency")
        print("          auto, manual, assisted")
        print("          goto x y z (e.g., 'goto 10 5 15')")
        print("          help - show this help")
        print("====================================\\n")
    
    def _input_worker(self):
        """Рабочий поток для текстового ввода."""
        while self.running:
            try:
                # Безопасный ввод без блокировки основного потока
                user_input = input("QIKI> ").strip().lower()
                if user_input:
                    self._process_text_command(user_input)
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                self.logger.error(f"Text input error: {e}")
    
    def _process_text_command(self, text: str):
        """Обрабатывает текстовую команду."""
        parts = text.split()
        if not parts:
            return
        
        command_name = parts[0]
        
        if command_name == 'help':
            self._show_help()
            return
        
        if command_name in self.text_commands:
            command_type, base_params = self.text_commands[command_name]
            params = base_params.copy()
            
            # Специальная обработка команды goto
            if command_name == 'goto' and len(parts) >= 4:
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    params['position'] = np.array([x, y, z])
                except ValueError:
                    print("Error: goto requires three numbers (x y z)")
                    return
            elif command_name == 'goto':
                print("Usage: goto x y z")
                return
            
            command = ControlCommand(
                command_type=command_type,
                parameters=params,
                source="text"
            )
            
            # Добавляем команду в очередь
            try:
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(
                    self.command_queue.put(command), loop
                )
                print(f"Command executed: {command_name}")
            except Exception as e:
                self.logger.error(f"Error queuing text command: {e}")
        else:
            print(f"Unknown command: {command_name}. Type 'help' for available commands.")
    
    def _show_help(self):
        """Показывает справку по командам."""
        print("\\nAvailable commands:")
        print("  Movement: forward, backward, left, right, up, down")
        print("  Actions:  hover, land, reset, stop, emergency")
        print("  Modes:    auto, manual, assisted")
        print("  Navigate: goto x y z")
        print("  System:   help")
        print()
    
    async def get_command(self) -> Optional[ControlCommand]:
        """Получает команду из очереди."""
        try:
            return await asyncio.wait_for(self.command_queue.get(), timeout=0.001)
        except asyncio.TimeoutError:
            return None
    
    def stop(self) -> None:
        """Останавливает интерфейс."""
        self.running = False


class AndroidSafeInterface(InputInterface):
    """
    Безопасный интерфейс для Android без root прав.
    Использует только стандартные методы ввода Python.
    """
    
    def __init__(self):
        self.logger = Logger.get_logger("android_safe_interface")
        self.command_queue = asyncio.Queue()
        self.running = False
        self.input_thread = None
        
        # Команды для Android (упрощенные)
        self.android_commands = {
            # Движение (простые команды)
            '1': (CommandType.MOVE_FORWARD, {}),
            '2': (CommandType.MOVE_BACKWARD, {}),
            '3': (CommandType.MOVE_LEFT, {}),
            '4': (CommandType.MOVE_RIGHT, {}),
            '5': (CommandType.MOVE_UP, {}),
            '6': (CommandType.MOVE_DOWN, {}),
            
            # Быстрые команды
            '7': (CommandType.HOVER, {}),
            '8': (CommandType.LAND, {}),
            '9': (CommandType.EMERGENCY_STOP, {}),
            '0': (CommandType.RESET_POSITION, {}),
            
            # Режимы (буквы)
            'a': (CommandType.CHANGE_MODE, {'mode': ControlMode.AUTONOMOUS}),
            'm': (CommandType.CHANGE_MODE, {'mode': ControlMode.MANUAL}),
            's': (CommandType.CHANGE_MODE, {'mode': ControlMode.ASSISTED}),
        }
    
    async def start(self) -> None:
        """Запускает Android-безопасный интерфейс."""
        self.running = True
        self.input_thread = threading.Thread(target=self._safe_input_worker, daemon=True)
        self.input_thread.start()
        self.logger.info("Android safe interface started")
        
        print("\\n🤖 QIKI Bot Control - Android Safe Mode (No Root Required)")
        print("==========================================================")
        print("🔢 Movement: 1=Forward 2=Back 3=Left 4=Right 5=Up 6=Down")
        print("⚡ Quick:    7=Hover 8=Land 9=STOP 0=Reset")
        print("🎮 Modes:    a=Auto m=Manual s=Assisted")
        print("✅ Root-free: Works on any Android device")
        print("==========================================================\\n")
    
    def _safe_input_worker(self):
        """Безопасный рабочий поток для Android."""
        while self.running:
            try:
                # Безопасный input() без системных вызовов
                user_input = input("QIKI-Android> ").strip().lower()
                
                if user_input in self.android_commands:
                    command_type, params = self.android_commands[user_input]
                    command = ControlCommand(
                        command_type=command_type,
                        parameters=params,
                        source="android_safe"
                    )
                    
                    # Неблокирующе добавляем команду в очередь
                    try:
                        loop = asyncio.get_event_loop()
                        asyncio.run_coroutine_threadsafe(
                            self.command_queue.put(command), loop
                        )
                        print(f"✅ Command: {command_type.name}")
                    except Exception as e:
                        self.logger.error(f"Error queuing Android command: {e}")
                
                elif user_input == 'help' or user_input == 'h':
                    self._show_help()
                
                elif user_input in ['exit', 'quit', 'q']:
                    break
                    
                elif user_input:
                    print(f"❌ Unknown command: {user_input}. Type 'help' for commands.")
                    
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                self.logger.error(f"Android safe input error: {e}")
    
    def _show_help(self):
        """Показывает справку по командам."""
        print("\\n📖 QIKI Android Commands Help")
        print("=============================")
        print("Movement Controls:")
        print("  1 - Move Forward    2 - Move Backward")
        print("  3 - Move Left       4 - Move Right")
        print("  5 - Move Up         6 - Move Down")
        print("\\nQuick Actions:")
        print("  7 - Hover in place  8 - Land")
        print("  9 - Emergency STOP  0 - Reset position")
        print("\\nControl Modes:")
        print("  a - Autonomous mode m - Manual mode")
        print("  s - Assisted mode")
        print("\\nSystem:")
        print("  help/h - Show this help")
        print("  exit/quit/q - Exit interface")
        print("=============================\\n")
    
    async def get_command(self) -> Optional[ControlCommand]:
        """Получает команду из очереди."""
        try:
            return await asyncio.wait_for(self.command_queue.get(), timeout=0.001)
        except asyncio.TimeoutError:
            return None
    
    def stop(self) -> None:
        """Останавливает интерфейс."""
        self.running = False


class BotController:
    """
    Главный контроллер бота - объединяет все интерфейсы ввода и управляет ботом.
    """
    
    def __init__(self, agent=None, physics=None):
        self.logger = Logger.get_logger("bot_controller")
        self.agent = agent
        self.physics = physics
        
        # Состояние управления
        self.control_state = ControlState()
        
        # Интерфейсы ввода
        self.interfaces: List[InputInterface] = []
        self.active_interfaces: List[InputInterface] = []
        
        # Очередь команд с приоритетами
        self.command_queue = asyncio.PriorityQueue()
        
        # Параметры управления
        self.movement_speed = config.AGENT_SPEED_FACTOR
        self.rotation_speed = config.AGENT_ROTATION_SPEED
        self.control_sensitivity = config.BOT_CONTROL_SENSITIVITY
        
        # Android-специфичные настройки
        self.android_no_root = config.ANDROID_NO_ROOT_MODE
        self.android_safe_input = config.ANDROID_SAFE_INPUT_MODE
        self.mobile_friendly = config.MOBILE_FRIENDLY_UI
        
        # История команд для отладки
        self.command_history = []
        
        self.logger.info("BotController initialized")
    
    def add_interface(self, interface: InputInterface):
        """Добавляет интерфейс ввода."""
        self.interfaces.append(interface)
        self.logger.info(f"Added interface: {interface.__class__.__name__}")
    
    async def start(self):
        """Запускает контроллер и все интерфейсы."""
        # Определяем доступные интерфейсы в зависимости от платформы
        self._detect_and_add_interfaces()
        
        # Запускаем все интерфейсы
        for interface in self.interfaces:
            try:
                await interface.start()
                self.active_interfaces.append(interface)
            except Exception as e:
                self.logger.warning(f"Failed to start interface {interface.__class__.__name__}: {e}")
        
        self.logger.info(f"BotController started with {len(self.active_interfaces)} active interfaces")
        
        # Запускаем основной цикл обработки команд
        asyncio.create_task(self._command_processing_loop())
    
    def _detect_and_add_interfaces(self):
        """Автоматически определяет и добавляет доступные интерфейсы."""
        # Определяем тип устройства и настройки
        is_android = self._is_android_environment()
        is_mobile = self._is_mobile_environment()
        is_desktop = self._is_desktop_environment()
        
        # Лог информации о платформе
        platform_info = []
        if is_android:
            platform_info.append("Android")
        if is_mobile:
            platform_info.append("Mobile")
        if is_desktop:
            platform_info.append("Desktop")
        
        self.logger.info(f"Platform detected: {', '.join(platform_info) if platform_info else 'Unknown'}")
        
        # Специальный Android-интерфейс для работы без root
        if is_android and self.android_no_root:
            try:
                self.add_interface(AndroidSafeInterface())
                self.logger.info("Added Android safe interface (no root required)")
            except Exception as e:
                self.logger.warning(f"Could not add Android safe interface: {e}")
        
        # Текстовый интерфейс (всегда доступен, но не на Android в безопасном режиме)
        if not (is_android and self.android_no_root):
            self.add_interface(TextInterface())
        
        # Клавиатурный интерфейс для десктопа (не на Android без root)
        if is_desktop and not (is_android and self.android_no_root):
            try:
                self.add_interface(KeyboardInterface())
            except Exception as e:
                self.logger.warning(f"Could not add keyboard interface: {e}")
        
        # Сенсорный интерфейс для мобильных устройств (не Android в безопасном режиме)
        if (is_mobile or is_android) and not (is_android and self.android_no_root):
            try:
                touch_interface = TouchInterface()
                if is_android:
                    # Устанавливаем безопасный режим для Android
                    touch_interface.set_touch_mode('directional')
                self.add_interface(touch_interface)
            except Exception as e:
                self.logger.warning(f"Could not add touch interface: {e}")
        
        # Fallback: если никакие интерфейсы не добавились, добавляем текстовый
        if not self.interfaces:
            self.logger.warning("No interfaces detected, adding text interface as fallback")
            self.add_interface(TextInterface())
        
        # Специальный интерфейс для Android без root
        if is_android and self.android_no_root:
            try:
                self.add_interface(AndroidSafeInterface())
            except Exception as e:
                self.logger.warning(f"Could not add Android safe interface: {e}")
    
    def _is_android_environment(self) -> bool:
        """Проверяет, работаем ли мы в среде Android."""
        return (
            'ANDROID_ROOT' in os.environ or 
            'ANDROID_DATA' in os.environ or
            'ANDROID_STORAGE' in os.environ or
            sys.platform.startswith('android') or
            os.path.exists('/system/build.prop') or
            os.path.exists('/system/app')
        )
    
    def _is_desktop_environment(self) -> bool:
        """Проверяет, работаем ли мы в десктопной среде."""
        try:
            # Проверяем наличие терминала с поддержкой клавиатуры
            if os.name == 'nt':  # Windows
                try:
                    import importlib.util
                    if importlib.util.find_spec("msvcrt"):
                        return True
                except ImportError:
                    pass
                return False
            else:  # Unix/Linux
                try:
                    import importlib.util
                    if (importlib.util.find_spec("termios") and 
                        importlib.util.find_spec("tty")):
                        # Проверяем, что stdin - это терминал (безопасно)
                        return hasattr(sys.stdin, 'fileno') and os.isatty(sys.stdin.fileno())
                except (ImportError, OSError):
                    pass
                return False
        except Exception as e:
            self.logger.debug(f"Desktop detection error: {e}")
            return False
    
    def _is_mobile_environment(self) -> bool:
        """Проверяет, работаем ли мы на мобильном устройстве."""
        # Расширенная эвристика для мобильных устройств
        mobile_indicators = [
            # Android
            'ANDROID_ROOT' in os.environ,
            'ANDROID_DATA' in os.environ,
            sys.platform.startswith('android'),
            # iOS (если когда-то понадобится)
            'IPHONEOS_DEPLOYMENT_TARGET' in os.environ,
            # Общие признаки мобильных
            'MOBILE' in os.environ.get('USER_AGENT', ''),
            # Пути характерные для мобильных
            os.path.exists('/sdcard'),
            os.path.exists('/system/framework'),
        ]
        
        return any(mobile_indicators)
    
    async def _command_processing_loop(self):
        """Основной цикл обработки команд."""
        while True:
            try:
                # Собираем команды от всех интерфейсов
                for interface in self.active_interfaces:
                    command = await interface.get_command()
                    if command:
                        # Добавляем в очередь с приоритетом
                        await self.command_queue.put((command.priority, time.time(), command))
                
                # Обрабатываем команды из очереди
                if not self.command_queue.empty():
                    _, _, command = await asyncio.wait_for(
                        self.command_queue.get(), timeout=0.001
                    )
                    await self._execute_command(command)
                
                # Короткая пауза
                await asyncio.sleep(0.01)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Command processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _execute_command(self, command: ControlCommand):
        """Выполняет команду управления."""
        self.logger.debug(f"Executing command: {command.command_type.name} from {command.source}")
        
        # Добавляем в историю
        self.command_history.append(command)
        if len(self.command_history) > 100:
            self.command_history.pop(0)
        
        try:
            if command.command_type == CommandType.EMERGENCY_STOP:
                await self._emergency_stop()
            
            elif command.command_type == CommandType.CHANGE_MODE:
                mode = command.parameters.get('mode', ControlMode.MANUAL)
                self._change_control_mode(mode)
            
            elif command.command_type in [
                CommandType.MOVE_FORWARD, CommandType.MOVE_BACKWARD,
                CommandType.MOVE_LEFT, CommandType.MOVE_RIGHT,
                CommandType.MOVE_UP, CommandType.MOVE_DOWN
            ]:
                await self._execute_movement_command(command)
            
            elif command.command_type in [
                CommandType.TURN_LEFT, CommandType.TURN_RIGHT,
                CommandType.PITCH_UP, CommandType.PITCH_DOWN,
                CommandType.ROLL_LEFT, CommandType.ROLL_RIGHT
            ]:
                await self._execute_rotation_command(command)
            
            elif command.command_type == CommandType.GOTO_POSITION:
                await self._goto_position(command.parameters.get('position'))
            
            elif command.command_type == CommandType.HOVER:
                await self._hover()
            
            elif command.command_type == CommandType.RESET_POSITION:
                await self._reset_position()
            
            else:
                self.logger.warning(f"Unhandled command type: {command.command_type.name}")
        
        except Exception as e:
            self.logger.error(f"Error executing command {command.command_type.name}: {e}")
    
    async def _execute_movement_command(self, command: ControlCommand):
        """Выполняет команду движения."""
        if not self.agent:
            return
        
        # Направления движения
        direction_map = {
            CommandType.MOVE_FORWARD: np.array([1.0, 0.0, 0.0]),
            CommandType.MOVE_BACKWARD: np.array([-1.0, 0.0, 0.0]),
            CommandType.MOVE_LEFT: np.array([0.0, 1.0, 0.0]),
            CommandType.MOVE_RIGHT: np.array([0.0, -1.0, 0.0]),
            CommandType.MOVE_UP: np.array([0.0, 0.0, 1.0]),
            CommandType.MOVE_DOWN: np.array([0.0, 0.0, -1.0]),
        }
        
        direction = direction_map.get(command.command_type, np.zeros(3))
        
        # Применяем скорость
        velocity = direction * self.movement_speed
        
        # Относительное или абсолютное движение
        if command.parameters.get('relative', False):
            # Для сенсорного управления - используем переданные скорости
            velocity[0] = command.parameters.get('velocity_x', velocity[0])
            velocity[1] = command.parameters.get('velocity_y', velocity[1])
        
        # Устанавливаем целевую скорость для агента
        if hasattr(self.agent, 'target_velocity'):
            self.agent.target_velocity = velocity
        
        self.control_state.thrust_vector = velocity
        self.control_state.last_command_time = time.time()
    
    async def _execute_rotation_command(self, command: ControlCommand):
        """Выполняет команду вращения."""
        if not self.agent:
            return
        
        # Направления вращения
        rotation_map = {
            CommandType.TURN_LEFT: np.array([0.0, 0.0, 1.0]),
            CommandType.TURN_RIGHT: np.array([0.0, 0.0, -1.0]),
            CommandType.PITCH_UP: np.array([0.0, 1.0, 0.0]),
            CommandType.PITCH_DOWN: np.array([0.0, -1.0, 0.0]),
            CommandType.ROLL_LEFT: np.array([1.0, 0.0, 0.0]),
            CommandType.ROLL_RIGHT: np.array([-1.0, 0.0, 0.0]),
        }
        
        rotation = rotation_map.get(command.command_type, np.zeros(3))
        angular_velocity = rotation * self.rotation_speed
        
        self.control_state.torque_vector = angular_velocity
        self.control_state.last_command_time = time.time()
    
    async def _goto_position(self, position: Optional[np.ndarray]):
        """Переходит к указанной позиции."""
        if position is not None and self.agent:
            self.agent.target = position
            self.control_state.target_position = position
            self.logger.info(f"Moving to position: {position}")
    
    async def _hover(self):
        """Зависает на текущей позиции."""
        if self.agent and self.physics:
            current_pos = self.physics.get_state().position
            self.agent.target = current_pos
            self.control_state.target_position = current_pos
            self.logger.info("Hovering at current position")
    
    async def _reset_position(self):
        """Сбрасывает позицию к началу координат."""
        reset_pos = np.array([0.0, 0.0, 5.0])  # 5 метров над землей
        await self._goto_position(reset_pos)
        self.logger.info("Position reset to origin")
    
    async def _emergency_stop(self):
        """Экстренная остановка."""
        self.control_state.emergency_stop = True
        if self.agent:
            self.agent.emergency_stop = True
        
        # Обнуляем все команды
        self.control_state.thrust_vector = np.zeros(3)
        self.control_state.torque_vector = np.zeros(3)
        
        self.logger.warning("EMERGENCY STOP ACTIVATED")
    
    def _change_control_mode(self, mode: ControlMode):
        """Меняет режим управления."""
        self.control_state.mode = mode
        if self.agent:
            self.agent.autonomous = (mode == ControlMode.AUTONOMOUS)
        
        self.logger.info(f"Control mode changed to: {mode.name}")
    
    def get_control_state(self) -> ControlState:
        """Возвращает текущее состояние управления."""
        return self.control_state
    
    def stop(self):
        """Останавливает контроллер."""
        for interface in self.active_interfaces:
            interface.stop()
        self.logger.info("BotController stopped")
