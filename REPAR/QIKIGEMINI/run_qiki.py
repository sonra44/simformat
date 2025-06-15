#!/usr/bin/env python3
import asyncio
import signal
import sys
import argparse

from main import QikiSimulation

async def shutdown(simulation):
    """Корректно завершает работу симуляции"""
    print("\nShutting down...")
    await simulation.stop()

async def run_qiki_simulation(enable_visualization=True):
    """
    Основная функция для запуска симуляции QIKI.
    """
    # Создаем экземпляр симуляции
    simulation = QikiSimulation(enable_visualization=enable_visualization)
    
    # Настраиваем обработку сигналов для корректного завершения
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(simulation)))
    
    try:
        print("\nQIKI Simulation Controls:")
        print("q - quit")
        print("p - pause/resume")
        print("d - toggle debug mode")
        print("r - reset")
        print("\nStarting simulation...")
        
        # Запускаем все асинхронные компоненты
        tasks = [
            # Основной цикл симуляции
            asyncio.create_task(simulation.run()),
            # Запускаем агента - это уже делается в simulation.run()
            # asyncio.create_task(simulation.agent.run_control_loop()),
            # Запускаем обновление сенсоров - это уже делается в simulation.run()
            # asyncio.create_task(simulation.sensors.start_updates())
        ]
        
        # Ждем завершения основной задачи
        await asyncio.gather(*tasks)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise
    finally:
        await shutdown(simulation)

def main():
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="QIKI Simulation")
    parser.add_argument("--no-vis", "--no-visualizer", action="store_true", help="Disable visualization")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    try:
        # Используем новый EventLoop политику для Windows
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Запускаем симуляцию
        asyncio.run(run_qiki_simulation(enable_visualization=not args.no_vis))
        
    except KeyboardInterrupt:
        pass  # Обработка Ctrl+C уже реализована через signal handlers
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
