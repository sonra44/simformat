#!/usr/bin/env python3
import sys
import traceback
import logging
import os
import asyncio

async def run_with_full_traceback():
    """Запускает run_qiki.py с полным отслеживанием ошибок"""
    print("Запуск QIKI симуляции с полным отслеживанием ошибок")
    
    try:
        # Настраиваем логирование для вывода всех уровней сообщений
        logging.basicConfig(level=logging.DEBUG, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler()])
        
        # Импортируем и запускаем функцию main из run_qiki
        print("Импортируем модуль run_qiki...")
        import run_qiki
        
        print("Запускаем main() из run_qiki...")
        await run_qiki.main()
        
    except Exception as e:
        print("\n\n==== ОШИБКА ЗАПУСКА СИМУЛЯЦИИ ====")
        print(f"Тип ошибки: {type(e).__name__}")
        print(f"Сообщение: {str(e)}")
        print("\nПолный стек вызовов:")
        traceback.print_exc()
        print("==== КОНЕЦ СТЕКА ОШИБКИ ====\n")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        # Для Windows используем правильную политику цикла событий
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Переходим в директорию QIKIGEMINI
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Запускаем с полным отслеживанием ошибок
        exit_code = asyncio.run(run_with_full_traceback())
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
        sys.exit(130)  # Стандартный код выхода для прерывания сигналом
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        traceback.print_exc()
        sys.exit(1)
