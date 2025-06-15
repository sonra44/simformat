import logging
import os
from datetime import datetime
import threading
from config import config  # Импортируем экземпляр конфигурации
import sys # Добавлен импорт sys

class Logger:
    """
    Синглтон-класс Logger для централизованного управления логированием в приложении.
    Гарантирует, что основной логгер "qiki" и его обработчики (файл, консоль)
    настраиваются только один раз.
    Предоставляет методы для получения именованных логгеров, которые наследуют
    конфигурацию от основного.
    """

    _instance = None
    _lock = threading.Lock()
    _initialized = False # Флаг инициализации, управляемый строго

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Logger, cls).__new__(cls)
                    # Вызываем _initialize_logger только один раз при создании экземпляра
                    # Он не будет вызываться через __init__
                    cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        """
        Инициализирует основной логгер и его обработчики.
        Вызывается только один раз при создании синглтон-экземпляра.
        """
        if self._initialized:
            return

        self.log_dir: str = "logs"
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path: str = os.path.join(self.log_dir, f"qiki_{timestamp}.log")

        # Получаем основной логгер для приложения
        self.root_logger = logging.getLogger("qiki")
        # Уровень логирования устанавливается из Config
        self.root_logger.setLevel(config.LOG_LEVEL)
        # Отключаем propagation, чтобы предотвратить дублирование, если есть другие корневые логгеры
        self.root_logger.propagate = False

        # Форматтер для всех обработчиков
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Обработчик для записи в файл
        self.file_handler = logging.FileHandler(self.log_file_path)
        self.file_handler.setLevel(config.LOG_LEVEL) # Уровень из Config
        self.file_handler.setFormatter(formatter)
        self.root_logger.addHandler(self.file_handler)

        # Обработчик для вывода в консоль
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(config.LOG_LEVEL) # Уровень из Config
        self.console_handler.setFormatter(formatter)
        self.root_logger.addHandler(self.console_handler)

        self._initialized = True
        self.root_logger.info(f"Central QIKI Logger initialized to level {logging.getLevelName(config.LOG_LEVEL)}. Log file: {self.log_file_path}")

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Возвращает именованный экземпляр logging.Logger.
        Всегда должен использоваться для получения логгеров в модулях.
        """
        # Гарантируем, что синглтон-логгер инициализирован
        if cls._instance is None:
            # При первом вызове, создаем и инициализируем синглтон
            _ = cls.__new__(cls) # Это вызовет _initialize_logger через __new__

        # Возвращаем стандартный logging.Logger, который наследует настройки от 'qiki'
        return logging.getLogger(f"qiki.{name}")

    def set_level(self, level: int) -> None:
        """
        Устанавливает уровень логирования для основного логгера и его обработчиков.
        """
        with self._lock:
            if not self._initialized:
                self.root_logger.warning("Attempted to set level on uninitialized logger.")
                return

            # Логируем изменение уровня до того, как основной логгер изменит свой уровень,
            # но после того, как изменятся уровни обработчиков,
            # чтобы сообщение прошло через текущий (более низкий) уровень.
            # Для надежности, сделаем это сообщение WARNING, чтобы оно не отфильтровывалось.
            self.root_logger.warning(f"Central QIKI Logger level set to {logging.getLevelName(level)}")

            self.root_logger.setLevel(level)
            self.file_handler.setLevel(level)
            self.console_handler.setLevel(level)

    def close(self) -> None:
        """
        Закрывает все обработчики логгера и очищает его состояние.
        Должен быть вызван явно перед завершением приложения.
        """
        with self._lock:
            if not self._initialized:
                # self.root_logger.debug("Logger already closed or not initialized.") # Нельзя логировать, если не инициализирован
                return

            # Логируем закрытие перед фактическим закрытием, используя базовый print
            # так как логгер может быть уже в процессе закрытия
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - qiki - INFO - Closing central QIKI Logger handlers.", file=sys.stderr)

            for handler in self.root_logger.handlers[:]: # Итерируем по копии списка
                try:
                    handler.close()
                    self.root_logger.removeHandler(handler)
                except Exception as e:
                    print(f"Error closing logger handler {handler}: {e}", file=sys.stderr)

            # Очистка состояния синглтона
            self._initialized = False
            Logger._instance = None # Сбросить экземпляр для возможности переинициализации (например, в тестах)