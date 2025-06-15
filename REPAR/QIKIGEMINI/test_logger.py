import os
import time
import logging
from config import Config
from logger import Logger # Импортируем наш Logger
import sys # Добавлен импорт sys
import io

# Временное переопределение уровня логирования для теста, если это необходимо
# config.LOG_LEVEL = logging.DEBUG

def run_logger_test_scenario():
    """
    Проверяет корректность работы синглтона Logger,
    отсутствие дублирования обработчиков и функциональность.
    """
    print("\n--- Запуск сценария тестирования Logger.py ---")

    # Временно устанавливаем уровень логирования DEBUG для теста
    original_log_level = config.LOG_LEVEL
    config.LOG_LEVEL = logging.DEBUG
    # Сбросить синглтон логгера перед запуском теста, чтобы он инициализировался с новым уровнем
    if Logger._instance is not None:
        Logger._instance.close()

    # Перехватываем stdout и stderr для проверки вывода в консоль
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_stdout = io.StringIO()
    sys.stderr = captured_stderr = io.StringIO()

    log_file_path = None
    try:
        # Шаг 1: Инициализация первого логгера (должен инициализировать синглтон)
        main_logger = Logger.get_logger("main_app")
        main_logger.info("Main application started.")
        main_logger.debug("This is a debug message from main_app.") # Теперь должно логироваться

        log_file_path = Logger._instance.log_file_path # Получаем путь к файлу логов из синглтона

        # Шаг 2: Получение другого логгера (не должен повторно инициализировать синглтон)
        module_logger = Logger.get_logger("module_a")
        module_logger.warning("Something potentially bad happened in module A.")
        module_logger.error("An error occurred in module A.")

        # Шаг 3: Получение третьего логгера
        another_logger = Logger.get_logger("another_module")
        another_logger.info("Info from another module.")

        # Шаг 4: Проверка, что все логгеры используют один и тот же основной логгер (qiki)
        # и имеют одни и те же обработчики
        assert main_logger.parent.name == "qiki"
        assert module_logger.parent.name == "qiki"
        assert another_logger.parent.name == "qiki"

        # Проверяем количество обработчиков у корневого логгера 'qiki'
        # Должно быть 2: FileHandler и StreamHandler
        root_qiki_logger = logging.getLogger("qiki")
        assert len(root_qiki_logger.handlers) == 2, \
            f"Ожидалось 2 обработчика, найдено: {len(root_qiki_logger.handlers)}"

        # Шаг 5: Проверка изменения уровня логирования
        main_logger.info(f"Current log level for main_app: {logging.getLevelName(main_logger.level)}")
        Logger._instance.set_level(logging.WARNING) # Изменяем уровень через синглтон
        main_logger.info("This info message should NOT appear after level change (level is WARNING).") # Должно быть проигнорировано
        module_logger.error("This error message should appear.") # Должно появиться

        # Шаг 6: Закрытие логгера
        Logger._instance.close()

        # Попытка логирования после закрытия (не должно вызывать ошибок, но и не должно логировать)
        main_logger.info("This message should not be logged after close.")


        # Проверяем, что хэндлеры были удалены
        assert len(root_qiki_logger.handlers) == 0, \
            f"Обработчики не были удалены: {len(root_qiki_logger.handlers)}"

        print("\n--- Тест Logger.py завершен. Проверяем вывод ---")

    finally:
        # Восстанавливаем stdout и stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        # Восстанавливаем оригинальный уровень логирования в Config
        config.LOG_LEVEL = original_log_level

        # Выводим захваченные данные
        print("\n--- Консольный вывод (stdout): ---")
        print(captured_stdout.getvalue())
        print("\n--- Консольный вывод (stderr): ---")
        print(captured_stderr.getvalue())

        # Проверка содержимого файла логов
        if log_file_path and os.path.exists(log_file_path):
            with open(log_file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
                print(f"\n--- Содержимое файла логов ({log_file_path}): ---")
                print(file_content)

            # Ассерты для файла логов
            assert "Main application started." in file_content
            assert "This is a debug message from main_app." in file_content # Теперь ожидаем, что будет
            assert "Something potentially bad happened in module A." in file_content
            assert "An error occurred in module A." in file_content
            assert "Info from another module." in file_content
            assert "This info message should NOT appear after level change (level is WARNING)." not in file_content # Должно быть отфильтровано
            assert "This error message should appear." in file_content
            assert "Central QIKI Logger initialized to level DEBUG" in file_content # Уровень теперь DEBUG
            assert "Central QIKI Logger level set to WARNING" in file_content
            assert "Closing central QIKI Logger handlers." in file_content # Это будет в stderr из-за print

            print("\n--- Проверки файла логов завершены успешно. ---")
            # Очистка: удалить файл логов после теста
            try:
                os.remove(log_file_path)
                print(f"Файл логов '{log_file_path}' удален.")
                # Удаляем директорию, если пуста
                if not os.listdir(os.path.dirname(log_file_path)): # Проверяем родительскую директорию
                    os.rmdir(os.path.dirname(log_file_path))
                    print(f"Директория логов '{os.path.dirname(log_file_path)}' удалена.")

            except Exception as e:
                print(f"Ошибка при очистке файла логов или директории: {e}", file=sys.stderr)
        else:
            print("\n--- Файл логов не найден или не был создан. ---")


if __name__ == "__main__":
    run_logger_test_scenario()