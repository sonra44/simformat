#!/usr/bin/env python3
"""
QIKI Simulation - Минимальная рабочая версия
==========================================
"""

import time
import sys
import numpy as np
import asyncio
import os

print("🤖 QIKI Bot Simulation - Minimal Version")
print("========================================")
print("Starting basic simulation loop...")

class SimpleBot:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 5.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.target = np.array([10.0, 10.0, 10.0])
        self.time = 0.0
        
    def update(self, dt):
        # Простая физика - движение к цели
        direction = self.target - self.position
        distance = np.linalg.norm(direction)
        
        if distance > 0.1:
            direction = direction / distance
            self.velocity = direction * 2.0  # 2 м/с к цели
        else:
            self.velocity = np.zeros(3)
            
        # Обновление позиции
        self.position += self.velocity * dt
        self.time += dt
        
        # Применяем гравитацию
        self.velocity[2] -= 9.81 * dt
        
    def get_status(self):
        return {
            'position': self.position,
            'velocity': self.velocity,
            'target': self.target,
            'distance_to_target': np.linalg.norm(self.target - self.position),
            'time': self.time
        }

async def main():
    bot = SimpleBot()
    dt = 1.0 / 60.0  # 60 FPS
    
    print("\\n🚀 Simulation started!")
    print("Press Ctrl+C to stop\\n")
    
    try:
        for step in range(300):  # 5 секунд симуляции
            bot.update(dt)
            status = bot.get_status()
            
            # Выводим статус каждые 30 кадров (0.5 сек)
            if step % 30 == 0:
                print(f"⏱️  Time: {status['time']:.1f}s")
                print(f"📍 Position: [{status['position'][0]:.1f}, {status['position'][1]:.1f}, {status['position'][2]:.1f}]")
                print(f"🎯 Target: [{status['target'][0]:.1f}, {status['target'][1]:.1f}, {status['target'][2]:.1f}]")
                print(f"📏 Distance: {status['distance_to_target']:.1f}m")
                print(f"💨 Velocity: [{status['velocity'][0]:.1f}, {status['velocity'][1]:.1f}, {status['velocity'][2]:.1f}] m/s")
                print("-" * 50)
            
            await asyncio.sleep(dt)
            
    except KeyboardInterrupt:
        print("\\n🛑 Simulation stopped by user")
    
    print("\\n✅ Simulation completed!")
    print(f"Final position: {bot.position}")
    print(f"Final time: {bot.time:.1f}s")

if __name__ == "__main__":
    asyncio.run(main())
