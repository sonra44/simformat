#!/usr/bin/env python3
"""
Простой тест системы управления ботом для Android
"""

import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_import():
    """Тест импорта конфигурации."""
    try:
        from config import config
        print("✅ Config imported successfully")
        
        # Проверяем новые параметры
        assert hasattr(config, 'AGENT_SPEED_FACTOR'), "AGENT_SPEED_FACTOR missing"
        assert hasattr(config, 'AGENT_ROTATION_SPEED'), "AGENT_ROTATION_SPEED missing"
        assert hasattr(config, 'ANDROID_NO_ROOT_MODE'), "ANDROID_NO_ROOT_MODE missing"
        
        print(f"✅ AGENT_SPEED_FACTOR: {config.AGENT_SPEED_FACTOR}")
        print(f"✅ ANDROID_NO_ROOT_MODE: {config.ANDROID_NO_ROOT_MODE}")
        return True
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False

def test_bot_control_import():
    """Тест импорта модуля управления ботом."""
    try:
        from bot_control import BotController, AndroidSafeInterface, ControlCommand, CommandType
        print("✅ Bot control imported successfully")
        
        # Проверяем, что классы доступны
        assert BotController is not None, "BotController not available"
        assert AndroidSafeInterface is not None, "AndroidSafeInterface not available"
        
        print("✅ All bot control classes available")
        return True
    except Exception as e:
        print(f"❌ Bot control import failed: {e}")
        return False

def test_android_interface():
    """Тест Android интерфейса."""
    try:
        from bot_control import AndroidSafeInterface
        
        interface = AndroidSafeInterface()
        print("✅ AndroidSafeInterface created successfully")
        
        # Проверяем основные свойства
        assert hasattr(interface, 'android_commands'), "android_commands missing"
        assert len(interface.android_commands) > 0, "No Android commands defined"
        
        print(f"✅ Android commands available: {len(interface.android_commands)}")
        return True
    except Exception as e:
        print(f"❌ Android interface test failed: {e}")
        return False

def test_platform_detection():
    """Тест определения платформы."""
    try:
        from bot_control import BotController
        
        controller = BotController()
        
        # Проверяем методы определения платформы
        android_detected = controller._is_android_environment()
        mobile_detected = controller._is_mobile_environment()
        desktop_detected = controller._is_desktop_environment()
        
        print(f"✅ Platform detection:")
        print(f"   Android: {android_detected}")
        print(f"   Mobile: {mobile_detected}")
        print(f"   Desktop: {desktop_detected}")
        
        return True
    except Exception as e:
        print(f"❌ Platform detection test failed: {e}")
        return False

def main():
    """Основная функция тестирования."""
    print("🤖 QIKI Bot Control - Android Compatibility Test")
    print("=" * 50)
    
    tests = [
        ("Config Import", test_config_import),
        ("Bot Control Import", test_bot_control_import),
        ("Android Interface", test_android_interface),
        ("Platform Detection", test_platform_detection),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
        
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Android compatibility ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
