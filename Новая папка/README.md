# QIKI Simulation

## Обзор

QIKI Simulation - это асинхронная платформа для моделирования автономного робота (бота) в симулированной среде. Система основана на полностью асинхронной архитектуре, позволяющей эффективно моделировать физику, сенсоры, энергетические системы и поведение агента в реальном времени.

## Содержание

- [Структура проекта](#структура-проекта)
- [Требования](#требования)
- [Установка](#установка)
- [Запуск симуляции](#запуск-симуляции)
- [Архитектура](#архитектура)
- [Диаграмма компонентов](#диаграмма-компонентов)
- [Особенности асинхронной архитектуры](#особенности-асинхронной-архитектуры)
- [Предотвращение Deadlock](#предотвращение-deadlock)
- [Обработка ошибок](#обработка-ошибок)
- [Конфигурация](#конфигурация)
- [Мониторинг производительности](#мониторинг-производительности)
- [Решение проблем](#решение-проблем)
- [Расширение системы](#расширение-системы)
- [Примеры использования API](#примеры-использования-api)
- [Тестирование](#тестирование)
- [Известные ограничения](#известные-ограничения)
- [FAQ](#faq)
- [Roadmap](#roadmap)
- [История версий](#история-версий)
- [Глоссарий](#глоссарий)
- [Вклад в проект](#вклад-в-проект)
- [Разработчики](#разработчики)
- [Лицензия](#лицензия)

## Структура проекта

```
QIKIGEMINI/
├── main.py                   # Основной класс симуляции и точка входа
├── run_qiki.py               # Удобный запуск с поддержкой аргументов командной строки
├── agent.py                  # Асинхронный агент принятия решений
├── sensors.py                # Система сбора данных с сенсоров
├── physics.py                # Физическая модель бота
├── environment.py            # Окружающая среда и взаимодействие с ней
├── qik_os.py                 # Операционная система бота
├── config.py                 # Конфигурационные параметры
├── logger.py                 # Централизованная система логирования
├── analyzer.py               # Анализатор и сбор метрик
├── visualizer.py             # Графический визуализатор (matplotlib)
├── ascii_visualizer.py       # ASCII-визуализатор в консоли
├── ascii_visualizer_adapter.py # Адаптер для асинхронной работы с визуализатором
├── bot_interface_impl.py     # Реализация интерфейса бота
├── hardware/                 # Аппаратные компоненты
│   ├── frame_core.py         # Моделирование физического каркаса
│   ├── power_systems.py      # Энергетические системы и батареи
│   ├── thermal_system.py     # Термическая система и регулирование температуры
│   └── frame_core.py         # Структурная целостность каркаса
├── data/                     # Директория для хранения данных сессий
└── logs/                     # Директория для хранения логов
```

## Требования

Для запуска QIKI Simulation требуются следующие зависимости:

```
numpy>=2.3.0
scipy>=1.15.3
matplotlib>=3.10.3
pandas>=2.3.0
```

Все зависимости указаны в файле `requirements.txt`.

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/qiki-simulation.git
cd qiki-simulation
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Запуск симуляции

### Базовый запуск

```bash
python run_qiki.py
```

### Запуск без визуализации (для быстрой отладки)

```bash
python run_qiki.py --no-vis
```

### Запуск напрямую через main.py

```bash
python main.py [--no-vis]
```

## Архитектура

QIKI Simulation основана на асинхронной архитектуре с использованием Python asyncio. Ключевые компоненты:

### Ядро симуляции (main.py)

Центральный компонент, управляющий всем жизненным циклом симуляции, инициализирует все подсистемы и координирует их взаимодействие.

### Физика (physics.py)

Моделирует физическое поведение бота, включая:
- Движение (позиция, скорость, ускорение)
- Вращение (с использованием кватернионов)
- Применение сил и моментов

### Сенсоры (sensors.py)

Асинхронная система сбора данных со всех подсистем:
- Навигационные данные (позиция, скорость, ускорение)
- Энергетические показатели (заряд батареи, потребление)
- Термические показатели (температура ядра и радиаторов)
- Структурные показатели (целостность каркаса)

### Агент (agent.py)

Интеллектуальный компонент, принимающий решения на основе данных сенсоров:
- Навигация к целевой точке
- Реакция на критические состояния
- Оптимизация энергопотребления

### Окружение (environment.py)

Моделирует внешнюю среду:
- Гравитация
- Солнечное излучение
- Физические границы мира

### Аппаратные модули (hardware/)

Моделируют физические компоненты бота:
- Энергетические системы (батареи, солнечные панели)
- Термические системы (нагрев, охлаждение)
- Каркас (структурная целостность)

### QikOS (qik_os.py)

Операционная система бота:
- Управление задачами
- Обработка команд
- Приоритезация действий

### Визуализация

Два типа визуализации:
- Графическая (matplotlib)
- ASCII (консольная)

## Диаграмма компонентов

Ниже представлена архитектурная диаграмма, показывающая взаимодействие между компонентами QIKI Simulation:

```
                                 +----------------+
                                 |     Config     |
                                 +----------------+
                                        |
                                        v
+-------------+    +---------------+    +---------------+    +--------------+
|  Environment |<-->|               |<-->|   Hardware    |<-->|   Sensors    |
+-------------+    |               |    +---------------+    +--------------+
                   |               |         |   |                 |
                   |  QikiSimulation|<--------|---|-----------------+
                   |               |         |   |                 |
+-------------+    |               |    +----v---v----+    +--------------+
|  Visualizer  |<-->|               |<-->|    Agent     |<-->|    QikOS     |
+-------------+    +---------------+    +--------------+    +--------------+
     |  |                  |
     |  |           +------v------+
     |  +---------->|   Logger    |
     |              +-------------+
     |
+----v---------+
|   Analyzer   |
+--------------+
```

Основные потоки данных:
1. **QikiSimulation** - центральный компонент, инициализирует и координирует все компоненты
2. **Physics → Sensors → Agent** - основной поток данных о состоянии системы
3. **Agent → Physics** - поток управляющих воздействий
4. **Environment → Physics** - внешние силы и воздействия
5. **Hardware → Sensors** - аппаратные метрики (энергия, температура)
6. **Config** - предоставляет параметры для всех компонентов
7. **Logger** - обеспечивает логирование со всех компонентов
8. **Visualizer/Analyzer** - отображают и анализируют данные моделирования

## Особенности асинхронной архитектуры

Система использует `async/await` для эффективного неблокирующего выполнения задач:

- **Основной цикл симуляции**: асинхронный метод `_main_loop()`
- **Обновление физики**: фиксированный временной шаг для стабильности
- **Сенсоры**: асинхронное чтение данных без блокировок
- **Агент**: асинхронное принятие решений
- **Визуализация**: адаптер для интеграции синхронного визуализатора с асинхронным кодом

## Предотвращение Deadlock

Важные моменты для предотвращения блокировок:

1. **Неблокирующее чтение сенсоров**: метод `_read_navigation` не вызывает блокирующее ожидание обновления физики
2. **События и очереди**: для координации компонентов используются события `asyncio.Event`
3. **Тайм-ауты**: все операции ожидания имеют тайм-ауты для предотвращения бесконечного ожидания
4. **Контроль доступа к ресурсам**: избегание циклических зависимостей

## Обработка ошибок

Система имеет многоуровневую обработку ошибок:

1. **Логирование**: централизованная система через `logger.py`
2. **Проверка критических состояний**: метод `sensors.check_critical_status()`
3. **Корректное завершение работы**: метод `shutdown()` в классе `QikiSimulation`

## Конфигурация

Все параметры симуляции настраиваются в `config.py`, включая:

- Физические параметры (масса, радиус, гравитация)
- Границы окружения
- Характеристики энергетических систем
- Термические параметры
- Настройки логирования

### Ключевые параметры конфигурации

| Параметр | Описание | Значение по умолчанию |
|----------|----------|------------------------|
| `TIME_STEP` | Временной шаг симуляции | 0.5 секунд |
| `MAX_TIME` | Максимальное время симуляции | 3600 секунд (1 час) |
| `BOT_MASS_INITIAL` | Начальная масса бота | 10.0 кг |
| `BOT_RADIUS` | Физический радиус бота | 0.5 м |
| `GRAVITY` | Вектор гравитации | [0.0, 0.0, -9.81] м/с² |
| `MAX_THRUST_N` | Максимальная сила тяги | 100.0 Н |
| `MAX_TORQUE_NM` | Максимальный крутящий момент | 50.0 Н·м |
| `BOUNDS` | Границы среды по осям | {"x": (-100, 100), "y": (-100, 100), "z": (-100, 100)} |
| `MAIN_BATTERY_CAPACITY_WH` | Ёмкость основной батареи | 10000 Вт·ч |
| `RESERVE_BATTERY_CAPACITY_WH` | Ёмкость резервной батареи | 2000 Вт·ч |
| `INITIAL_CORE_TEMPERATURE_C` | Начальная температура ядра | 20.0 °C |
| `THERMAL_RADIATION_COEFFICIENT` | Коэффициент теплового излучения | 0.8 |

## Мониторинг производительности

QIKI Simulation предоставляет несколько способов мониторинга производительности:

### Встроенные средства мониторинга

1. **Логирование производительности**
   ```python
   # Включение детального логирования производительности
   Logger.get_logger("main").set_level(logging.DEBUG)
   ```

2. **Мониторинг через визуализатор**
   - Графический визуализатор отображает FPS (кадры в секунду) в реальном времени
   - ASCII-визуализатор показывает время обновления в режиме DEBUG

3. **Анализатор симуляции**
   ```python
   # После запуска получить статистику производительности
   analyzer_stats = simulation.analyzer.calculate_session_statistics()
   print(f"Physics update time (avg): {analyzer_stats['physics_update_time_avg']} ms")
   ```

### Профилирование с внешними инструментами

Для более глубокого профилирования используйте:

```bash
# Профилирование CPU
python -m cProfile -o profile.stats run_qiki.py
# Анализ результатов
python -m pstats profile.stats

# Профилирование памяти с помощью pympler
pip install pympler
```

Затем добавьте в код:
```python
from pympler import tracker
memory_tracker = tracker.SummaryTracker()
# ... после запуска симуляции
memory_tracker.print_diff()
```

## Решение проблем

### Ошибка: "'State' object has no attribute 'time'"

Проблема связана с адаптером визуализатора, который ожидает атрибут 'time' в объекте State. Решение:

```python
# В ascii_visualizer_adapter.py, метод update должен принимать current_time:
def update(self, state: State, target_position: np.ndarray, sensor_data: Dict[str, Any], current_time: float = 0.0)

# В main.py передавайте current_time:
self.ascii_visualizer.update(state, self.agent.target, await self.sensors.read_all(), self.current_time)
```

### Ошибка: "Analyzer object has no attribute 'record_step_data'"

Правильный метод для записи данных в анализатор - `log_step`:

```python
self.analyzer.log_step(
    step=int(self.current_time * 100),
    current_sim_time=self.current_time,
    physics_obj=self.physics,
    sensors_data=await self.sensors.read_all(),
    agent_obj=self.agent
)
```

### Проблема: Deadlock между сенсорами и физикой

Если симуляция зависает, проверьте метод `_read_navigation` в `sensors.py`. Он не должен содержать вызовов `await self.physics_obj.wait_for_update()`, так как это создает циклическую зависимость.

### Проблема: Выход за границы окружения

Проверьте ключи в `config.BOUNDS`. Они должны быть `"x"`, `"y"`, `"z"`, а не `"min_x"`, `"max_x"` и т.д.

## Расширение системы

### Добавление нового сенсора

1. Добавьте метод чтения в класс `Sensors` (например, `_read_new_sensor`)
2. Включите его вызов в метод `read_all()`
3. Добавьте соответствующие проверки в `check_critical_status()`

### Добавление нового аппаратного модуля

1. Создайте новый класс в директории `hardware/`
2. Добавьте его инициализацию в `QikiSimulation.__init__()`
3. Интегрируйте с сенсорами и агентом

### Изменение поведения агента

Модифицируйте методы в классе `Agent`, особенно:
- `_calculate_control_forces()` - для изменения навигации
- `_evaluate_overall_status()` - для изменения реакции на состояния

## Примеры использования API

### Настройка и запуск симуляции программно

```python
import asyncio
from main import QikiSimulation

async def custom_simulation():
    # Создание экземпляра симуляции
    sim = QikiSimulation(enable_visualization=True)
    
    # Настройка агента
    sim.agent.set_target(np.array([5.0, 10.0, 15.0]))
    sim.agent.autonomous = True
    
    # Запуск симуляции
    await sim.run()
    
    # Получение результатов
    results = sim.analyzer.get_results()
    return results

# Запуск
results = asyncio.run(custom_simulation())
```

### Взаимодействие с работающей симуляцией

```python
# Изменение целевой точки во время работы
async def change_target(simulation):
    await asyncio.sleep(10)  # Ждем 10 секунд
    simulation.agent.set_target(np.array([20.0, 20.0, 20.0]))
    print("Target changed")

# Внесение изменений в окружающую среду
async def modify_environment(simulation):
    # Изменение гравитации (например, посадка на Марс)
    mars_gravity = np.array([0.0, 0.0, -3.72])
    simulation.environment.gravity = mars_gravity
    
    # Добавление солнечной активности
    simulation.environment.set_solar_activity(1500)  # W/m²
```

### Доступ к телеметрии

```python
# Получение текущего состояния
state = simulation.physics.get_state()
position = state.position
velocity = state.velocity

# Получение данных сенсоров
async def monitor_power():
    while simulation.running:
        sensor_data = await simulation.sensors.read_all()
        battery_percentage = sensor_data['power']['total_battery_percentage']
        print(f"Battery: {battery_percentage:.1f}%")
        await asyncio.sleep(5)
```

## Тестирование

### Запуск встроенных тестов

```bash
# Запуск всех тестов
python -m unittest discover tests

# Запуск конкретного теста
python -m unittest tests.test_physics
```

### Тестирование компонентов

1. **Физика**:
   ```python
   from physics import PhysicsObject
   import numpy as np
   
   # Создание объекта
   physics = PhysicsObject(name="Test", mass=10.0)
   
   # Применение силы
   physics.apply_force(np.array([10.0, 0.0, 0.0]))
   
   # Обновление на один шаг
   physics.update(dt=0.1)
   
   # Проверка состояния
   state = physics.get_state()
   assert state.velocity[0] > 0, "Velocity should increase after force applied"
   ```

2. **Сенсоры**:
   ```python
   from sensors import Sensors
   
   # Проверка критических состояний
   sensor_data = {
       "power": {"total_battery_percentage": 5.0},
       "thermal": {"core_temperature_c": 85.0},
       "navigation": {"position": [1000, 0, 0]}
   }
   
   critical_status = sensors.check_critical_status(sensor_data)
   assert critical_status["power_critical"], "Should detect low battery"
   assert critical_status["thermal_critical"], "Should detect high temperature"
   assert critical_status["out_of_bounds"], "Should detect out of bounds"
   ```

3. **Интеграционное тестирование**:
   ```python
   # Проверка взаимодействия компонентов
   async def test_agent_sensor_interaction():
       sim = QikiSimulation(enable_visualization=False)
       sim.agent.set_target(np.array([10.0, 0.0, 0.0]))
       
       # Запуск на короткое время
       sim._running = True
       await sim._update_simulation(dt=0.1)
       
       # Проверка, что агент получает данные сенсоров
       assert sim.agent.last_sensor_data is not None
       
       # Проверка, что сенсоры видят позицию из физики
       sensor_data = await sim.sensors.read_all()
       assert np.allclose(sensor_data["navigation"]["position"], 
                         sim.physics.get_state().position)
   ```

## Разработчики

- [Ваше имя] - Основной разработчик

## Лицензия

Этот проект распространяется под лицензией [указать лицензию].

## Известные ограничения

1. **Производительность при большом количестве объектов**
   - Текущая реализация оптимизирована для одного бота. Моделирование нескольких ботов может привести к существенному снижению производительности.
   - Рекомендуется не более 3-5 ботов в одной симуляции.

2. **Физика столкновений**
   - Обнаружение и обработка столкновений реализованы в упрощенном виде.
   - Для сложных взаимодействий между объектами могут возникать неточности.

3. **Масштабируемость визуализации**
   - Графический визуализатор на matplotlib может замедляться при длительной работе из-за накопления данных.
   - Рекомендуется периодически перезапускать визуализатор при длительных симуляциях.

4. **Ограничения точности физики**
   - При очень малых временных шагах (< 0.001с) или больших скоростях (> 1000 м/с) возможны численные нестабильности.
   - Если нужна высокоточная симуляция, рекомендуется использовать scipy.integrate.solve_ivp.

5. **Зависимость от SciPy для полной функциональности**
   - Без SciPy возможна работа в режиме совместимости, но с ограниченной точностью вращений.

## FAQ

### Общие вопросы

**Q: Как изменить границы моделируемого мира?**  
A: Измените параметр `BOUNDS` в файле `config.py`.

**Q: Можно ли использовать QIKI для моделирования нескольких ботов?**  
A: Да, но с ограничениями. Вам потребуется создать несколько экземпляров PhysicsObject и адаптировать Environment для их взаимодействия.

**Q: Какой временной шаг оптимален для стабильной симуляции?**  
A: Рекомендуется использовать значения от 0.01 до 0.05 секунд. Слишком маленький шаг увеличит нагрузку на CPU, слишком большой может привести к нестабильности физики.

### Технические вопросы

**Q: Симуляция зависает при старте. Что делать?**  
A: Наиболее вероятная причина - deadlock между компонентами. Проверьте асинхронные вызовы в `sensors.py` и `physics.py`. Убедитесь, что нет циклических зависимостей.

**Q: Почему не отображается ASCII-визуализатор?**  
A: Проверьте, что в методе update адаптера правильно передается current_time. Также убедитесь, что консоль поддерживает Unicode.

**Q: Как уменьшить потребление памяти во время длительных симуляций?**  
A: Уменьшите значения `TELEMETRY_HISTORY_LENGTH` и `PLOT_HISTORY` в config.py. Также можно добавить периодическое очищение анализатора через `analyzer.clear_old_data()`.

**Q: Ошибка импорта модулей. Как исправить?**  
A: Убедитесь, что вы запускаете скрипты из корневой директории проекта. Если проблема не решается, проверьте правильность структуры каталогов и установку всех зависимостей.

**Q: Как изменить поведение бота при низком заряде батареи?**  
A: Модифицируйте метод `_evaluate_power_status()` в классе Agent и добавьте соответствующую логику в `decide()`.

## Roadmap

### Краткосрочные планы (ближайшие 3 месяца)
- [ ] Улучшение обработки столкновений
- [ ] Добавление более сложных моделей окружающей среды (ветер, температурные зоны)
- [ ] Расширение возможностей анализатора для экспорта данных
- [ ] Оптимизация производительности при большом количестве объектов

### Среднесрочные планы (3-6 месяцев)
- [ ] Интеграция с машинным обучением для агента
- [ ] Разработка 3D-визуализатора на базе OpenGL/PyGame
- [ ] Добавление сетевых возможностей для распределенной симуляции
- [ ] Создание редактора сценариев с GUI

### Долгосрочные планы (6-12 месяцев)
- [ ] Поддержка мультиагентного моделирования
- [ ] Интеграция с ROS (Robot Operating System)
- [ ] Создание модуля для симуляции в реальном оборудовании
- [ ] Расширение на другие физические домены (жидкости, деформируемые тела)

## История версий

### v0.9.0 (Текущая)
- Полностью асинхронная архитектура
- Устранение deadlock между компонентами
- Исправление ошибок в адаптере визуализации
- Обновление документации

### v0.8.5
- Добавление ASCII-визуализатора
- Улучшение обработки ошибок
- Интеграция QikOS

### v0.8.0
- Базовая реализация физики, сенсоров и агента
- Первая версия визуализации
- Логирование и анализ данных

## Глоссарий

**Bot (QIKI-бот)** - Автономный робот, моделируемый в симуляции.

**QikOS** - Операционная система бота, управляющая задачами и ресурсами.

**Agent** - Интеллектуальный компонент, принимающий решения на основе данных сенсоров.

**Sensor** - Компонент, собирающий данные о состоянии бота и окружающей среды.

**Physics** - Компонент, моделирующий физическое поведение бота в окружающей среде.

**Environment** - Окружающая среда, с которой взаимодействует бот.

**Frame Core** - Структурный каркас бота, моделирующий его физическую целостность.

**Power Systems** - Энергетические системы бота, включая батареи и солнечные панели.

**Thermal System** - Система терморегуляции бота.

**Deadlock** - Ситуация взаимной блокировки компонентов, приводящая к "зависанию" симуляции.

## Вклад в проект

Мы приветствуем вклад в развитие QIKI Simulation! Вот как вы можете помочь:

### Процесс внесения изменений

1. **Форкните репозиторий** на GitHub
2. **Создайте ветку** для ваших изменений (`git checkout -b feature/amazing-feature`)
3. **Внесите изменения** и протестируйте их
4. **Зафиксируйте изменения** (`git commit -m 'Add amazing feature'`)
5. **Отправьте изменения** в ваш форк (`git push origin feature/amazing-feature`)
6. **Создайте Pull Request** в основной репозиторий

### Стандарты кода

- Следуйте стилю PEP 8
- Пишите документацию для всех публичных методов (docstrings)
- Добавляйте типизацию с помощью аннотаций типов
- Включайте тесты для новой функциональности

### Тестирование вашего кода

Перед отправкой Pull Request убедитесь, что ваш код:
- Проходит все существующие тесты (`python -m unittest discover tests`)
- Не создает новых предупреждений или ошибок linter'а
- Хорошо документирован
