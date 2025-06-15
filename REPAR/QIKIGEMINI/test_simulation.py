#!/usr/bin/env python3
import asyncio
import traceback
from main import QikiSimulation

async def test_simulation():
    print("Creating QikiSimulation instance...")
    sim = QikiSimulation()
    print("QikiSimulation instance created successfully")
    
    print("\nStarting simulation for 5 seconds...")
    try:
        # Запускаем симуляцию на 5 секунд
        run_task = asyncio.create_task(sim.run())
        
        # Ждем 5 секунд
        await asyncio.sleep(5)
        
        # Останавливаем симуляцию
        print("\nStopping simulation...")
        await sim.shutdown()
        
        # Ждем завершения задачи
        try:
            await run_task
        except asyncio.CancelledError:
            print("Simulation task was cancelled as expected")
            
        print("Simulation test completed successfully")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        traceback.print_exc()
        
        # Пытаемся остановить симуляцию в случае ошибки
        try:
            await sim.shutdown()
        except Exception as shutdown_error:
            print(f"Error during shutdown: {shutdown_error}")

if __name__ == "__main__":
    print("Starting QIKI test simulation...")
    asyncio.run(test_simulation())
    print("Test complete.")
