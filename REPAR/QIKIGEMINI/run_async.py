#!/usr/bin/env python3
import asyncio
import sys
import os

# Добавляем текущий каталог в путь импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main_sync import QikiSimulation

async def main():
    simulation = QikiSimulation()
    await simulation.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
