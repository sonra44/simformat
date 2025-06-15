import time
import sys
import numpy as np
import asyncio
import os  # Добавлен импорт os
from collections import deque # Импортируем deque напрямую
from typing import Dict, Any, Deque # Добавляем импорт Dict, Any, Deque


from config import config
from physics import PhysicsObject
from agent import Agent
from sensors import Sensors
from environment import Environment
from logger import Logger # Используем наш кастомный Logger
from analyzer import Analyzer
from hardware.frame_core import FrameCore
from hardware.power_systems import PowerSystems # type: ignore
from hardware.thermal_system import ThermalSystem
from qik_os import QikOS
from qiki_display_adapter import QikiDisplayAdapter
from qiki_bot_control_integration import QikiBotControlIntegration
from bot_interface_impl import QikiBotInterface


class QikiSimulation:
    """
    Основной класс симуляции QIKI.
    Инициализирует все компоненты, управляет циклом симуляции и завершением работы.
    """

    def __init__(self, enable_visualization=True):
        """
        Инициализирует симуляцию QIKI.
        
        Args:
            enable_visualization (bool): Флаг, включающий или отключающий визуализацию
        """
        # Создаем директории для логов и данных
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Инициализируем логгер
        self.logger = Logger.get_logger("main")
        self.logger.info("Initializing QIKI Simulation...")
        
        # Состояние симуляции
        self._running = False
        self._paused = False
        self.current_time = 0.0
        self.max_time = config.MAX_TIME  # Используем MAX_TIME вместо MAX_SIMULATION_TIME
        self.last_real_time = 0.0
        self._simulation_task = None
        self.enable_visualization = enable_visualization
        
        # Инициализация аппаратных компонентов
        self.frame_core = FrameCore()
        self.power_systems = PowerSystems()
        self.thermal_system = ThermalSystem()

        initial_bot_mass = (
            config.BOT_MASS_INITIAL # Базовая масса из конфига
            + self.frame_core.get_total_mass()
            + self.power_systems.get_total_mass()
            # + self.thermal_system.get_total_mass() # Если есть, но пока не суммируем
        )
        self.physics = PhysicsObject(name="QIKI_Physics", mass=initial_bot_mass)
        self.agent = Agent()
        self.sensors = Sensors(
            self.physics, self.frame_core, self.power_systems, self.thermal_system
        )
        self.environment = Environment()

        # Инициализация QikOS и BotInterface
        self.bot_interface = QikiBotInterface(
            physics_obj=self.physics,
            sensors_obj=self.sensors,
            agent_obj=self.agent,
            frame_core_obj=self.frame_core,
            power_systems_obj=self.power_systems,
            thermal_system_obj=self.thermal_system,
            environment_obj=self.environment,
            qiki_simulation_instance=self # Передаем ссылку на себя для доступа ко времени
        )
        self.qik_os = QikOS(self.bot_interface)


        # Инициализация визуализатора
        self.display_adapter = None
        
        # Инициализация системы управления ботом
        self.bot_control_integration = None
        
        if self.enable_visualization:
            # Инициализация нового эталонного дисплея
            try:
                self.display_adapter = QikiDisplayAdapter(qiki_simulation_instance=self)
                self.logger.info("QikiDisplayAdapter initialized.")
            except Exception as e:
                self.logger.warning(f"Failed to initialize display adapter: {e}")
                
            # Инициализация системы управления ботом
            try:
                self.bot_control_integration = QikiBotControlIntegration(
                    agent=self.agent,
                    physics=self.physics,
                    simulation=self
                )
                self.logger.info("QikiBotControlIntegration initialized.")
            except Exception as e:
                self.logger.warning(f"Failed to initialize bot control: {e}")
        else:
            self.logger.info("Visualization disabled.")
        
        # Инициализация анализатора для сбора данных симуляции
        self.analyzer = Analyzer()  # Инициализация анализатора

        # Используем ранее инициализированные переменные вместо создания новых
        self.time_step: float = config.TIME_STEP  # Временной шаг симуляции
        self.real_time_mode: bool = config.REAL_TIME_MODE  # Режим реального времени
        
        # Флаги для управления состоянием симуляции
        self._shutdown_called: bool = False  # Флаг вызова shutdown

        # Деки для хранения истории данных телеметрии для визуализатора
        # Используем те же ключи, что и в QikOS.get_aggregated_sensor_data()
        self.telemetry_data: Dict[str, Deque[Any]] = {
            "time": deque(maxlen=config.TELEMETRY_HISTORY_LENGTH),
            "position_x": deque(maxlen=config.TELEMETRY_HISTORY_LENGTH),
            "position_y": deque(maxlen=config.TELEMETRY_HISTORY_LENGTH),
            "position_z": deque(maxlen=config.TELEMETRY_HISTORY_LENGTH),
            "velocity_magnitude": deque(maxlen=config.TELEMETRY_HISTORY_LENGTH),
            "acceleration_magnitude": deque(maxlen=config.TELEMETRY_HISTORY_LENGTH),
            "battery_percentage": deque(maxlen=config.TELEMETRY_HISTORY_LENGTH),
            "core_temperature_c": deque(maxlen=config.TELEMETRY_HISTORY_LENGTH),
            "structural_integrity": deque(maxlen=config.TELEMETRY_HISTORY_LENGTH),
            "stress_level": deque(maxlen=config.TELEMETRY_HISTORY_LENGTH),
            "solar_irradiance": deque(maxlen=config.TELEMETRY_HISTORY_LENGTH),
        }

        # Регистрация обработчиков сигналов
        # signal.signal(signal.SIGINT, self._handle_signal)
        # signal.signal(signal.SIGTERM, self._handle_signal)
        self.logger.info("QIKI Simulation initialized.")

    async def run(self) -> None:
        """
        Запускает основной цикл симуляции.
        """
        if self._running:
            self.logger.warning("Simulation is already running.")
            return

        self.logger.info("Starting QIKI Simulation main loop...")
        self._running = True
        self.last_real_time = time.time()

        try:
            # Инициализируем дисплей
            if self.display_adapter:
                await self.display_adapter.start()
            
            # Инициализируем систему управления ботом
            if self.bot_control_integration:
                await self.bot_control_integration.start()
            
            # Создаем задачу для основного цикла
            print("Creating simulation tasks...")
            self.logger.info("Creating simulation tasks...")
            self._simulation_task = asyncio.create_task(self._main_loop())
            
            # Запускаем асинхронные компоненты
            tasks = [
                self._simulation_task,
                asyncio.create_task(self.agent.run_control_loop()),
                asyncio.create_task(self.sensors.start_updates())
            ]
            
            # Ждем завершения всех задач
            print(f"Waiting for {len(tasks)} tasks to complete...")
            self.logger.info(f"Waiting for {len(tasks)} tasks to complete...")
            
            # Ожидаем задачи с таймаутом, чтобы видеть прогресс
            for i in range(60):  # 60 секунд максимум
                done, pending = await asyncio.wait(tasks, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
                if done:
                    print(f"Tasks completed: {len(done)}, pending: {len(pending)}")
                    self.logger.info(f"Tasks completed: {len(done)}, pending: {len(pending)}")
                    for task in done:
                        try:
                            result = task.result()
                            print(f"Task result: {result}")
                            self.logger.info(f"Task result: {result}")
                        except Exception as e:
                            print(f"Task failed with error: {str(e)}")
                            self.logger.error(f"Task failed with error: {str(e)}", exc_info=True)
                    break
                else:
                    print(f"Still waiting for tasks... Time elapsed: {i}s")
                    self.logger.info(f"Still waiting for tasks... Time elapsed: {i}s")
            
            # Если задачи все еще выполняются, продолжаем ждать без таймаута
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
                
        except asyncio.CancelledError:
            self.logger.info("Simulation was cancelled.")
            print("Simulation was cancelled.")
        except KeyboardInterrupt:
            self.logger.info("Simulation interrupted by user.")
            print("Simulation interrupted by user.")
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            print(f"Simulation error: {e}")
        finally:
            try:
                await self.shutdown()
            except Exception as e:
                self.logger.critical(f"Critical error during shutdown: {e}", exc_info=True)
                print(f"Critical error during shutdown: {e}")
                sys.exit(1)

    async def _main_loop(self) -> None:
        """
        Основной асинхронный цикл симуляции.
        """
        fixed_dt = 1.0 / config.SIMULATION_RATE
        accumulated_time = 0.0
        last_time = time.perf_counter()
        iteration_count = 0

        self.logger.info("Entering main simulation loop...")
        print("Starting main loop...")

        try:
            while self.current_time < self.max_time and self._running and iteration_count < 300:  # Увеличиваем количество итераций до 300
                iteration_count += 1
                print(f"Main loop iteration {iteration_count}, time: {self.current_time:.2f}s, accumulated time: {accumulated_time:.5f}s")
                
                try:
                    current_time = time.perf_counter()
                    frame_time = current_time - last_time
                    last_time = current_time
                    
                    accumulated_time += frame_time
                    
                    # Добавляем периодический вывод состояния
                    if int(self.current_time) % 1 == 0 and frame_time > 0:  # Каждую секунду
                        self.logger.info(f"Simulation time: {self.current_time:.2f}s, dt: {frame_time:.5f}s")

                    # Fixed timestep update
                    print(f"Before physics updates, accumulated time: {accumulated_time:.5f}s, fixed_dt: {fixed_dt:.5f}s")
                    update_count = 0
                    while accumulated_time >= fixed_dt and update_count < 10:  # Ограничиваем количество обновлений физики в одной итерации
                        update_count += 1
                        print(f"Running physics update {update_count}")
                        
                        try:
                            print(f"Updating simulation at t={self.current_time:.2f}s, dt={fixed_dt:.5f}s")
                            await self._update_simulation(fixed_dt)
                            accumulated_time -= fixed_dt
                            print(f" - Updated time to {self.current_time:.2f}s")
                        except Exception as e:
                            self.logger.error(f"Error in _update_simulation: {str(e)}", exc_info=True)
                            print(f"Error in _update_simulation: {str(e)}")
                            break
                        
                    if update_count > 0 and int(self.current_time) % 1 == 0:
                        self.logger.info(f"Performed {update_count} physics updates")
                        print(f"Performed {update_count} physics updates")

                    # Проверка критических состояний и управление остановкой
                    try:
                        print("Reading sensor data for critical status check")
                        sensor_data = await self.sensors.read_all()
                        critical_status = self.sensors.check_critical_status(sensor_data)
                        if any(critical_status.values()):
                            self.logger.critical(f"Critical system status detected: {critical_status}. Initiating emergency shutdown.")
                            print(f"Critical system status detected: {critical_status}")
                            await self.qik_os.shell.execute_command("emergency_stop", priority=0) # Отправляем команду аварийной остановки
                            await self.shutdown(exit_code=1) # Завершаем симуляцию с кодом ошибки
                    except Exception as e:
                        self.logger.error(f"Error checking critical status: {str(e)}", exc_info=True)
                        print(f"Error checking critical status: {str(e)}")

                    # Проверка, не был ли запрошен останов извне (например, через сигнал)
                    if not self._running:
                        self.logger.info("Shutdown requested, exiting main loop.")
                        print("Shutdown requested, exiting main loop.")
                        break # Выход из цикла, если симуляция остановлена

                    # Для отладки: принудительно выходим после нескольких итераций
                    if iteration_count >= 50:  # Увеличиваем до 50 итераций
                        print(f"Debug: Exiting main loop after {iteration_count} iterations")
                        self.logger.info(f"Debug: Exiting main loop after {iteration_count} iterations")
                        self._running = False
                        break

                    # Короткая пауза для снижения нагрузки CPU
                    print("Sleeping for 0.001s")
                    await asyncio.sleep(0.001)
                    print("Resumed after sleep")
                    
                except Exception as e:
                    self.logger.error(f"Error in main loop iteration: {str(e)}", exc_info=True)
                    print(f"Error in main loop iteration: {str(e)}")
                    await asyncio.sleep(0.1)  # Пауза перед следующей попыткой
            
            print(f"Main loop finished after {iteration_count} iterations")
            self.logger.info(f"QIKI Simulation main loop finished at time {self.current_time:.2f}s.")
        except Exception as e:
            self.logger.critical(f"CRITICAL ERROR in main loop: {str(e)}", exc_info=True)
            print(f"CRITICAL ERROR in main loop: {str(e)}")
        finally:
            # Для отладки: проверяем, была ли симуляция остановлена корректно
            if self._running:
                print("Warning: Main loop exited but simulation is still running")
                self._running = False
            await self.shutdown() # Завершаем работу после окончания симуляции


    async def _update_telemetry_data(self) -> None:
        """Собирает и обновляет данные телеметрии для визуализатора."""
        state = self.physics.get_state()
        sensor_data = await self.sensors.read_all()

        self.telemetry_data["time"].append(self.current_time)
        self.telemetry_data["position_x"].append(state.position[0])
        self.telemetry_data["position_y"].append(state.position[1])
        self.telemetry_data["position_z"].append(state.position[2])
        self.telemetry_data["velocity_magnitude"].append(np.linalg.norm(state.velocity))
        self.telemetry_data["acceleration_magnitude"].append(np.linalg.norm(state.acceleration))
        self.telemetry_data["battery_percentage"].append(sensor_data.get("power", {}).get("total_battery_percentage", 0.0))
        self.telemetry_data["core_temperature_c"].append(sensor_data.get("thermal", {}).get("core_temperature_c", 0.0))
        self.telemetry_data["structural_integrity"].append(sensor_data.get("frame", {}).get("integrity_percentage", 100.0))
        self.telemetry_data["stress_level"].append(sensor_data.get("frame", {}).get("stress_level", 0.0))
        self.telemetry_data["solar_irradiance"].append(sensor_data.get("power", {}).get("solar_irradiance_w_m2", 0.0))


    def _transform_analyzer_data_for_visualizer(self) -> Dict[str, Deque[Any]]:
        """
        Трансформирует собранные анализатором данные в формат, подходящий
        для визуализатора (deque's).
        """
        # Этот метод не используется напрямую, так как telemetry_data обновляется в реальном времени.
        # Однако, он показывает, как можно было бы перевести данные из анализатора.
        transformed_data: Dict[str, Deque[Any]] = {
            "time": deque(),
            "position_x": deque(),
            "position_y": deque(),
            "position_z": deque(),
            "velocity_magnitude": deque(),
            "acceleration_magnitude": deque(),
            "battery_percentage": deque(),
            "core_temperature_c": deque(),
            "structural_integrity": deque(),
            "stress_level": deque(),
            "solar_irradiance": deque(),
        }

        for step in self.analyzer.session_data["steps"]:
            transformed_data["time"].append(step["simulation_time"])
            transformed_data["position_x"].append(step["physics_state"]["position"][0])
            transformed_data["position_y"].append(step["physics_state"]["position"][1])
            transformed_data["position_z"].append(step["physics_state"]["position"][2])
            transformed_data["velocity_magnitude"].append(np.linalg.norm(step["physics_state"]["velocity"]))
            transformed_data["acceleration_magnitude"].append(np.linalg.norm(step["physics_state"]["acceleration"]))
            transformed_data["battery_percentage"].append(step["sensor_data"]["power"]["total_battery_percentage"])
            transformed_data["core_temperature_c"].append(step["sensor_data"]["thermal"]["core_temperature_c"])
            transformed_data["structural_integrity"].append(step["frame_core_status"]["structural_integrity"])
            transformed_data["stress_level"].append(step["frame_core_status"]["sensors_data"]["stress_level"])
            transformed_data["solar_irradiance"].append(step["sensor_data"]["power"]["solar_irradiance_w_m2"])
        return transformed_data


    async def shutdown(self, exit_code: int = 0) -> None:
        """
        Осуществляет корректное завершение работы симуляции.
        """
        if self._shutdown_called:
            self.logger.info("Shutdown already initiated. Skipping duplicate call.")
            return

        self._shutdown_called = True
        self.logger.info("Initiating graceful shutdown of QIKI Simulation...")
        self._running = False # Останавливаем основной цикл

        # Отменяем основную задачу симуляции, если она запущена
        if self._simulation_task and not self._simulation_task.done():
            self.logger.info("Cancelling simulation main loop task...")
            self._simulation_task.cancel()
            try:
                await self._simulation_task
            except asyncio.CancelledError:
                self.logger.info("Simulation main loop task cancelled successfully.")
            except Exception as e:
                self.logger.error(f"Error while waiting for simulation task to cancel: {e}")

        # Останавливаем QikOS
        self.logger.info("Stopping QikOS...")
        await self.qik_os.stop()

        # Завершение работы визуализатора
        self.logger.info("Closing display...")
        if self.display_adapter:
            self.display_adapter.close()
        
        # Завершение работы системы управления
        self.logger.info("Stopping bot control...")
        if self.bot_control_integration:
            self.bot_control_integration.stop()

        # Финализация анализатора (сохранение отчета)
        self.logger.info("Finalizing analyzer data...")
        self.analyzer.finalize()

        self.logger.info(f"QIKI Simulation shutdown complete with exit code {exit_code}.")
        # sys.exit(exit_code) # Выход из приложения будет управляться main_entry_point

    # --- Обработчик сигналов (для Ctrl+C) ---
    # Переделаем, чтобы быть async-safe и совместимым с asyncio
    # Этот метод будет вызываться из внешнего обработчика сигналов
    def request_shutdown(self, signum, frame):
        self.logger.warning(f"Signal {signum} received. Requesting graceful shutdown...")
        # Запускаем асинхронную функцию shutdown в текущем event loop
        # Это безопасно, так как мы находимся вне основного цикла, но в контексте event loop
        if self._simulation_task and not self._simulation_task.done():
            # Это может быть вызвано только один раз
            asyncio.create_task(self.shutdown(exit_code=0))
        else:
            self.logger.info("Simulation task not active or already shutting down.")
            # Если симуляция еще не запущена или уже завершается, просто завершаем логгер
            if self.main_logger_instance and hasattr(self.main_logger_instance, 'close') and self.main_logger_instance._initialized:
                self.main_logger_instance.close()
            sys.exit(0)

    def _calculate_power_consumption(self) -> float:
        """
        Рассчитывает общее потребление энергии для всех систем
        
        Returns:
            float: Общее потребление энергии в ваттах
        """
        # Базовое потребление электроэнергии
        power_consumption = config.POWER_CONSUMPTION_BASE_W
        
        # Добавляем потребление при движении
        # Используем physics.net_force вместо physics.get_state().net_force
        if np.linalg.norm(self.physics.net_force) > 0.1:
            power_consumption += config.POWER_CONSUMPTION_PROPULSION_ION_W
            
        # Добавляем потребление при активном агенте
        if self.agent.autonomous:
            power_consumption += config.POWER_CONSUMPTION_COMPUTATION_W
            
        # Добавляем накладные расходы QikOS
        power_consumption += config.POWER_CONSUMPTION_QIKOS_OVERHEAD_W
        
        return power_consumption

    async def _update_simulation(self, dt: float) -> None:
        """
        Асинхронное обновление всех компонентов симуляции
        
        Args:
            dt: Временной шаг симуляции
        """
        try:
            # 1. Обновление времени
            self.current_time += dt
            
            # 2. Обновление физики и окружения
            print(" - Applied environmental forces")
            await self.environment.apply_environmental_forces(self.physics)
            print(" - Updated physics")
            await self.physics.update(dt)
            
            # 3. Обновление аппаратных систем
            state = self.physics.get_state()
            print(" - Updated frame core")
            self.frame_core.update(state.acceleration, state.angular_acceleration)
            
            # 4. Обновление энергетических систем
            power_consumption = self._calculate_power_consumption()
            solar_irradiance = self.environment.get_solar_irradiance(state.position)
            print(" - Updated power systems")
            self.power_systems.update(dt, power_consumption, solar_irradiance)
            
            # 5. Обновление термальной системы
            heat_generated = power_consumption * config.EFFICIENCY_LOSS_HEAT_FACTOR
            print(" - Updated thermal system")
            self.thermal_system.update(dt, heat_generated)
            
            # 6. Обработка команд агента
            if self.agent.autonomous and not self.agent.emergency_stop:
                print(" - Reading sensor data for agent")
                sensor_data = await self.sensors.read_all()
                print(" - Updating agent")
                await self.agent.update(sensor_data)
            
            # 7. Обновление телеметрии
            print(" - Updating telemetry")
            await self._update_telemetry_data()
            
            # 8. Обновление визуализации
            try:
                if self.display_adapter:
                    print(" - Updating display adapter")
                    self.display_adapter.update(state, self.agent.target, await self.sensors.read_all(), self.current_time)
            except Exception as e:
                self.logger.error(f"Error updating visualization: {str(e)}", exc_info=True)
                print(f"Error updating visualization: {str(e)}")
            
            # 9. Запись данных в анализатор
            try:
                print(" - Recording analyzer data")
                self.analyzer.log_step(
                    step=int(self.current_time * 100),  # Шаг как целое число (для уникальности)
                    current_sim_time=self.current_time,
                    physics_obj=self.physics,
                    sensors_data=await self.sensors.read_all(),
                    agent_obj=self.agent
                )
                print(" - Analyzer data recorded")
            except Exception as e:
                self.logger.error(f"Error recording analyzer data: {str(e)}", exc_info=True)
                print(f"Error recording analyzer data: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error in _update_simulation: {str(e)}", exc_info=True)
            print(f"Error in _update_simulation: {str(e)}")
            raise  # Повторно вызываем исключение для обработки в вызывающем коде

    async def stop(self) -> None:
        """
        Останавливает симуляцию.
        Метод-обертка для совместимости с интерфейсом остановки из run_qiki.py
        """
        self.logger.info("Stopping simulation via stop() method...")
        await self.shutdown()
        
if __name__ == "__main__":
    import asyncio
    import argparse
    
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="QIKI Simulation")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    args = parser.parse_args()
    
    # Создаем экземпляр симуляции
    sim = QikiSimulation(enable_visualization=not args.no_vis)
    
    # Запускаем симуляцию
    print("Starting QIKI simulation...")
    try:
        asyncio.run(sim.run())
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    except Exception as e:
        print(f"Error running simulation: {e}")
        raise
