#!/usr/bin/env python3
"""
Тест системы управления ботом QIKI
Проверяет работоспособность всех интерфейсов управления
"""

import asyncio
import sys
import os
import time
import numpy as np

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot_control import BotController, ControlCommand, CommandType, ControlMode
from qiki_bot_control_integration import QikiBotControlIntegration
from logger import Logger


class MockAgent:
    """Мок-объект агента для тестирования."""
    def __init__(self):
        self.target = np.array([0.0, 0.0, 5.0])
        self.autonomous = True
        self.emergency_stop = False


class MockPhysics:
    """Мок-объект физики для тестирования."""
    def __init__(self):
        self.position = np.array([0.0, 0.0, 5.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.mass = 10.0
        self.forces = []
        self.torques = []
    
    def get_state(self):
        class State:
            def __init__(self, pos, vel):
                self.position = pos
                self.velocity = vel
        return State(self.position, self.velocity)
    
    def apply_force(self, force):
        self.forces.append(force)
        print(f"Applied force: {force}")
    
    def apply_torque(self, torque):
        self.torques.append(torque)
        print(f"Applied torque: {torque}")


async def test_bot_controller():
    """Тестирует базовую функциональность BotController."""
    print("🧪 Testing BotController...")
    
    # Создаем мок-объекты
    mock_agent = MockAgent()
    mock_physics = MockPhysics()
    
    # Создаем контроллер
    controller = BotController(agent=mock_agent, physics=mock_physics)
    
    try:
        # Запускаем контроллер
        await controller.start()
        print("✅ BotController started successfully")
        
        # Тестируем различные команды
        test_commands = [
            ControlCommand(CommandType.MOVE_FORWARD, source="test"),
            ControlCommand(CommandType.TURN_LEFT, source="test"),
            ControlCommand(CommandType.HOVER, source="test"),
            ControlCommand(CommandType.GOTO_POSITION, {"position": np.array([10, 5, 15])}, source="test"),
        ]
        
        for cmd in test_commands:
            await controller.command_queue.put((cmd.priority, time.time(), cmd))
            print(f"✅ Queued command: {cmd.command_type.name}")
        
        # Даем время на обработку команд
        await asyncio.sleep(1.0)
        
        # Проверяем состояние
        state = controller.get_control_state()
        print(f"✅ Control state: {state.mode.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ BotController test failed: {e}")
        return False
    finally:
        controller.stop()


async def test_integration():
    """Тестирует интеграцию с симуляцией."""
    print("🧪 Testing QikiBotControlIntegration...")
    
    # Создаем мок-объекты
    mock_agent = MockAgent()
    mock_physics = MockPhysics()
    
    # Создаем интеграцию
    integration = QikiBotControlIntegration(
        agent=mock_agent, 
        physics=mock_physics, 
        simulation=None
    )
    
    try:
        # Запускаем интеграцию
        await integration.start()
        print("✅ Integration started successfully")
        
        # Проверяем статус
        status = integration.get_control_status()
        print(f"✅ Control status: {status}")
        
        # Даем время на инициализацию
        await asyncio.sleep(2.0)
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    finally:
        integration.stop()


def test_platform_detection():
    """Тестирует определение платформы."""
    print("🧪 Testing platform detection...")
    
    try:
        # Создаем контроллер для тестирования
        controller = BotController()
        
        # Тестируем определение платформы
        is_desktop = controller._is_desktop_environment()
        is_mobile = controller._is_mobile_environment()
        
        print(f"✅ Desktop environment: {is_desktop}")
        print(f"✅ Mobile environment: {is_mobile}")
        
        # Проверяем переменные окружения Android
        android_vars = ['ANDROID_ROOT', 'ANDROID_DATA']
        android_detected = any(var in os.environ for var in android_vars)
        print(f"✅ Android environment variables: {android_detected}")
        
        # Проверяем sys.platform
        print(f"✅ Platform: {sys.platform}")
        
        return True
        
    except Exception as e:
        print(f"❌ Platform detection test failed: {e}")
        return False


def test_command_types():
    """Тестирует все типы команд."""
    print("🧪 Testing command types...")
    
    try:
        # Проверяем все типы команд
        command_types = list(CommandType)
        print(f"✅ Available command types: {len(command_types)}")
        
        for cmd_type in command_types:
            print(f"  - {cmd_type.name}")
        
        # Создаем тестовые команды
        test_commands = [
            ControlCommand(CommandType.MOVE_FORWARD),
            ControlCommand(CommandType.EMERGENCY_STOP, priority=0),
            ControlCommand(CommandType.GOTO_POSITION, {"position": np.array([1, 2, 3])}),
        ]
        
        for cmd in test_commands:
            print(f"✅ Created command: {cmd.command_type.name}, priority: {cmd.priority}")
        
        return True
        
    except Exception as e:
        print(f"❌ Command types test failed: {e}")
        return False


async def main():
    """Главная функция тестирования."""
    print("🚀 QIKI Bot Control System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Platform Detection", test_platform_detection),
        ("Command Types", test_command_types),
        ("Bot Controller", test_bot_controller),
        ("Integration", test_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\\n📋 Running: {test_name}")
        print("-" * 30)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Результаты
    print("\\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\\n📈 Summary: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Bot control system is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n💥 Test suite crashed: {e}")
        sys.exit(1)
