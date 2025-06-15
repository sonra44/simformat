import numpy as np
from config import config
from physics import PhysicsObject, State
from environment import Environment
from logger import Logger

async def update(self, dt):  # Всё ещё асинхронный метод
    import numpy as np
    from config import config
    from physics import PhysicsObject, State
    from environment import Environment
    from logger import Logger

    def main():
        # Инициализация логгера
        logger = Logger.get_logger("debug_test")
        logger.info("Starting debug test")

        try:
            # Создаем базовые компоненты
            env = Environment()
            logger.info("Environment created successfully")

            initial_state = State(
                position=np.array([0.0, 0.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0])
            )
            physics = PhysicsObject(
                name="DEBUG_BOT",
                mass=config.BOT_MASS_INITIAL,
                initial_state=initial_state
            )
            logger.info("Physics object created successfully")

            # Проверяем базовые параметры
            logger.info(f"Bot radius: {config.BOT_RADIUS}")
            logger.info(f"Gravity vector: {config.GRAVITY}")
            logger.info(f"Environment bounds: {config.BOUNDS}")

            # Тестируем базовую физику
            env.apply_environmental_forces(physics)
            logger.info("Environmental forces applied successfully")

            return True

        except Exception as e:
            logger.error(f"Debug test failed: {str(e)}")
            return False

    if __name__ == "__main__":
        success = main()
        print("Debug test completed successfully" if success else "Debug test failed")