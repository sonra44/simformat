# project_schema.py

"""
Эталонная схема проекта для системы верификации кода.
Определяет ожидаемую структуру, зависимости и архитектурные требования проекта.
Служит "контрактом" между компонентами системы и верификатором.
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field  # Добавлено field для default_factory
from enum import Enum


class ComponentType(Enum):
    """Типы компонентов в системе."""

    CORE = "CORE"  # Ядро системы
    HARDWARE = "HARDWARE"  # Аппаратные компоненты
    SENSOR = "SENSOR"  # Сенсоры
    INTERFACE = "INTERFACE"  # Интерфейсы
    UTILITY = "UTILITY"  # Утилиты
    VISUALIZATION = "VISUALIZATION"  # Визуализация
    TEST = "TEST"  # Тесты


class ImportType(Enum):
    """Типы импортов в системе."""

    STANDARD = "STANDARD"  # Стандартные библиотеки Python
    THIRD_PARTY = "THIRD_PARTY"  # Сторонние библиотеки
    LOCAL = "LOCAL"  # Локальные модули проекта


@dataclass
class ImportRequirement:
    """Требования к импорту."""

    module: str
    import_type: ImportType
    aliases: List[str] = field(default_factory=list)
    is_required: bool = True


@dataclass
class MethodRequirement:
    """Требование к методу класса."""

    name: str
    required: bool = True
    parameters: Optional[List[str]] = field(default_factory=list)  # Используем field
    return_type: Optional[str] = None
    docstring_required: bool = False
    is_classmethod: bool = False  # Добавлено для @classmethod


@dataclass
class ClassRequirement:
    """Требование к классу."""

    name: str
    methods: List[MethodRequirement]
    required: bool = True
    inheritance: Optional[List[str]] = field(default_factory=list)  # Используем field
    docstring_required: bool = True


@dataclass
class ModuleRequirement:
    """Требование к модулю."""

    name: str
    component_type: ComponentType
    imports: List[ImportRequirement]
    classes: List[ClassRequirement]
    functions: Optional[List[str]] = field(default_factory=list)  # Используем field
    config_usage: Set[str] = field(default_factory=set)  # Используем field
    file_pattern: Optional[str] = None


@dataclass
class ComponentRequirement:
    """Требования к компоненту."""

    name: str
    component_type: ComponentType
    required_methods: Set[str] = field(default_factory=set)
    required_attributes: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)


@dataclass
class ConfigUsage:
    """Требования к использованию конфигурации."""

    config_keys: Set[str]
    is_required: bool = True
    validation_rules: Dict[str, str] = field(default_factory=dict)


@dataclass
class ValidationRule:
    """Правило валидации."""

    rule_type: str
    severity: str
    condition: str
    message: str
    fix_suggestion: Optional[str] = None


# Определяем требования к модулям
MODULE_REQUIREMENTS = {
    "main": ComponentRequirement(
        name="QikiSimulation",
        component_type=ComponentType.CORE,
        required_methods={"__init__", "run", "shutdown"},
        required_attributes={"physics_obj", "environment", "agent"},
    ),
    "physics": ComponentRequirement(
        name="PhysicsObject",
        component_type=ComponentType.CORE,
        required_methods={"update", "apply_force", "apply_torque"},
        required_attributes={"position", "velocity", "acceleration"},
    ),
    "environment": ComponentRequirement(
        name="Environment",
        component_type=ComponentType.CORE,
        required_methods={"apply_environmental_forces"},
        required_attributes={"gravity", "bounds", "bot_radius"},
    ),
}

# Требования к использованию конфигурации
CONFIG_USAGE_REQUIREMENTS = {
    "environment": ConfigUsage(
        {
            "GRAVITY",
            "BOUNDS",
            "BOT_RADIUS",
            "ENVIRONMENT_DRAG_COEFFICIENT",
        }
    ),
    "physics": ConfigUsage(
        {
            "BOT_MASS_INITIAL",
            "GRAVITY",
            "THRUSTER_FORCE",
            "TORQUE_MAX",
            "MAX_VELOCITY",
        }
    ),
}

# Архитектурные правила
ARCHITECTURE_RULES = [
    ValidationRule(
        rule_type="dependency",
        severity="ERROR",
        condition="config должен быть импортирован во все модули кроме logger",
        message="Модуль должен использовать параметры из config.py",
        fix_suggestion="Добавьте 'from config import Config' в начало файла",
    ),
    ValidationRule(
        rule_type="inheritance",
        severity="ERROR",
        condition="все классы с interface в названии должны быть абстрактными",
        message="Интерфейсные классы должны быть абстрактными",
        fix_suggestion="Добавьте декоратор @abstractmethod к методам интерфейса",
    ),
]

# Исключения для валидации
VALIDATION_EXCEPTIONS = {
    "test_*.py": ["unused-import", "missing-docstring"],
    "logger.py": ["config-import"],
}

# Схема импортов с детализацией
IMPORT_REQUIREMENTS = {
    "main": [
        ImportRequirement("time", ImportType.STANDARD),
        ImportRequirement("sys", ImportType.STANDARD),
        ImportRequirement("signal", ImportType.STANDARD),
        ImportRequirement("numpy", ImportType.THIRD_PARTY, aliases=["np"]),
        ImportRequirement("config", ImportType.LOCAL),
        ImportRequirement("physics", ImportType.LOCAL),
        ImportRequirement("agent", ImportType.LOCAL),
        ImportRequirement("sensors", ImportType.LOCAL),
        ImportRequirement("environment", ImportType.LOCAL),
        ImportRequirement("visualizer", ImportType.LOCAL),
        ImportRequirement("logger", ImportType.LOCAL),
        ImportRequirement("analyzer", ImportType.LOCAL),
        ImportRequirement("hardware.frame_core", ImportType.LOCAL),
        ImportRequirement("hardware.power_systems", ImportType.LOCAL),
        ImportRequirement("hardware.thermal_system", ImportType.LOCAL),
    ],
    "agent": [
        ImportRequirement(
            "numpy", ImportType.THIRD_PARTY, aliases=["np"]
        ),  # Добавлен alias
        ImportRequirement("time", ImportType.STANDARD),
        ImportRequirement("logger", ImportType.LOCAL),
        ImportRequirement("config", ImportType.LOCAL),
    ],
    "analyzer": [
        ImportRequirement("json", ImportType.STANDARD),
        ImportRequirement(
            "numpy", ImportType.THIRD_PARTY, aliases=["np"]
        ),  # Добавлен alias
        ImportRequirement("os", ImportType.STANDARD),
        ImportRequirement(
            "datetime", ImportType.STANDARD, aliases=["datetime"]
        ),  # Явный alias
        ImportRequirement("logger", ImportType.LOCAL),
        ImportRequirement("config", ImportType.LOCAL),
    ],
    "config": [
        ImportRequirement(
            "numpy", ImportType.THIRD_PARTY, aliases=["np"]
        ),  # Добавлен alias
    ],
    "environment": [
        ImportRequirement(
            "numpy", ImportType.THIRD_PARTY, aliases=["np"]
        ),  # Добавлен alias
        ImportRequirement("config", ImportType.LOCAL),
    ],
    "frame_core": [
        ImportRequirement(
            "numpy", ImportType.THIRD_PARTY, aliases=["np"]
        ),  # Добавлен alias
        ImportRequirement("config", ImportType.LOCAL),
        ImportRequirement("logger", ImportType.LOCAL),
    ],
    "logger": [
        ImportRequirement("logging", ImportType.STANDARD),
        ImportRequirement("os", ImportType.STANDARD),
        ImportRequirement(
            "datetime", ImportType.STANDARD, aliases=["datetime"]
        ),  # Явный alias
        ImportRequirement("threading", ImportType.STANDARD),
    ],
    "physics": [
        ImportRequirement(
            "numpy", ImportType.THIRD_PARTY, aliases=["np"]
        ),  # Добавлен alias
        ImportRequirement(
            "scipy.spatial.transform", ImportType.THIRD_PARTY
        ),  # dataclasses убран
        ImportRequirement("logger", ImportType.LOCAL),
        ImportRequirement("config", ImportType.LOCAL),
        ImportRequirement("sys", ImportType.STANDARD),  # Добавлен, т.к. используется
    ],
    "power_systems": [
        ImportRequirement(
            "numpy", ImportType.THIRD_PARTY, aliases=["np"]
        ),  # Добавлен alias
        ImportRequirement("config", ImportType.LOCAL),
        ImportRequirement("logger", ImportType.LOCAL),
    ],
    "sensors": [
        ImportRequirement(
            "numpy", ImportType.THIRD_PARTY, aliases=["np"]
        ),  # Добавлен alias
        ImportRequirement("config", ImportType.LOCAL),
        ImportRequirement("logger", ImportType.LOCAL),
    ],
    "thermal_system": [
        ImportRequirement(
            "numpy", ImportType.THIRD_PARTY, aliases=["np"]
        ),  # Добавлен alias
        ImportRequirement("config", ImportType.LOCAL),
        ImportRequirement("logger", ImportType.LOCAL),
    ],
    "visualizer": [
        ImportRequirement("matplotlib.pyplot", ImportType.THIRD_PARTY, aliases=["plt"]),
        ImportRequirement(
            "mpl_toolkits.mplot3d", ImportType.THIRD_PARTY
        ),  # Убедимся, что он есть
        ImportRequirement(
            "numpy", ImportType.THIRD_PARTY, aliases=["np"]
        ),  # Добавлен alias
        ImportRequirement("collections", ImportType.STANDARD),
        ImportRequirement("config", ImportType.LOCAL),
        ImportRequirement("time", ImportType.STANDARD),
        ImportRequirement("logger", ImportType.LOCAL),  # Добавлен, т.к. используется
    ],
    "code_verifier": [
        ImportRequirement("os", ImportType.STANDARD),
        ImportRequirement("ast", ImportType.STANDARD),
        ImportRequirement("sys", ImportType.STANDARD),
        ImportRequirement("json", ImportType.STANDARD),  # Добавлен, т.к. используется
        ImportRequirement(
            "traceback", ImportType.STANDARD
        ),  # Добавлен, т.к. используется
        ImportRequirement("pathlib", ImportType.STANDARD),
        ImportRequirement("typing", ImportType.STANDARD),
        ImportRequirement(
            "dataclasses", ImportType.STANDARD
        ),  # Добавлен, т.к. используется
        ImportRequirement("enum", ImportType.STANDARD),  # Добавлен, т.к. используется
        ImportRequirement(
            "collections", ImportType.STANDARD
        ),  # Добавлен, т.к. используется
        ImportRequirement("project_schema", ImportType.LOCAL),
    ],
}

# Детализированная схема компонентов
MODULE_REQUIREMENTS = {
    "main": ModuleRequirement(
        name="main",
        component_type=ComponentType.CORE,
        imports=IMPORT_REQUIREMENTS["main"],
        classes=[
            ClassRequirement(
                name="QikiSimulation",
                methods=[
                    MethodRequirement("__init__", parameters=["self"]),
                    MethodRequirement(
                        "handle_signal", parameters=["self", "signum", "_frame"]
                    ),
                    MethodRequirement("run", parameters=["self"]),
                    MethodRequirement("shutdown", parameters=["self"]),
                    MethodRequirement("cleanup", parameters=["self"]),
                ],
            )
        ],
    ),
    "agent": ModuleRequirement(
        name="agent",
        component_type=ComponentType.CORE,
        imports=IMPORT_REQUIREMENTS["agent"],
        classes=[
            ClassRequirement(
                name="Agent",
                methods=[
                    MethodRequirement("__init__", parameters=["self"]),
                    MethodRequirement(
                        "set_autonomous_mode", parameters=["self", "mode"]
                    ),
                    MethodRequirement("set_target", parameters=["self", "new_target"]),
                    MethodRequirement(
                        "decide",
                        parameters=[
                            "self",
                            "current_time",
                            "physics_obj",
                            "sensor_data",
                        ],
                    ),
                    MethodRequirement(
                        "_evaluate_overall_status",
                        parameters=[
                            "self",
                            "integrity_percentage",
                            "total_battery_percentage",
                            "core_temperature_c",
                            "stress_level",
                        ],
                    ),
                    MethodRequirement(
                        "_evaluate_temperature_status",
                        parameters=["self", "core_temperature_c"],
                    ),
                    MethodRequirement(
                        "_navigate_to_target",
                        parameters=[
                            "self",
                            "current_position",
                            "current_velocity",
                            "current_forward_vector",
                        ],
                    ),
                    MethodRequirement("set_random_target", parameters=["self"]),
                    MethodRequirement(
                        "set_emergency_stop", parameters=["self", "status"]
                    ),
                    MethodRequirement("get_status", parameters=["self"]),
                ],
            )
        ],
    ),
    "analyzer": ModuleRequirement(
        name="analyzer",
        component_type=ComponentType.UTILITY,
        imports=IMPORT_REQUIREMENTS["analyzer"],
        classes=[
            ClassRequirement(
                name="Analyzer",
                methods=[
                    MethodRequirement("__init__", parameters=["self"]),
                    MethodRequirement(
                        "log_step",
                        parameters=[
                            "self",
                            "step",
                            "current_sim_time",  # Updated
                            "physics_obj",
                            "sensors_data",  # Updated
                            "agent_obj",  # Updated
                        ],
                    ),
                    MethodRequirement(
                        "calculate_session_statistics", parameters=["self"]
                    ),  # Исправлено с generate_report
                    MethodRequirement("finalize", parameters=["self"]),
                ],
            )
        ],
    ),
    "config": ModuleRequirement(
        name="config",
        component_type=ComponentType.CORE,
        imports=IMPORT_REQUIREMENTS["config"],
        classes=[
            ClassRequirement(
                name="Config",
                methods=[],  # Config - это класс с атрибутами, а не методами
                docstring_required=False,  # Обычно для Config docstring на уровне модуля
            )
        ],
    ),
    "environment": ModuleRequirement(
        name="environment",
        component_type=ComponentType.CORE,
        imports=IMPORT_REQUIREMENTS["environment"],
        classes=[
            ClassRequirement(
                name="Environment",
                methods=[
                    MethodRequirement("__init__", parameters=["self"]),
                    MethodRequirement(  # Updated name
                        "apply_environmental_forces", parameters=["self", "physics_obj"]
                    ),
                    MethodRequirement(  # Updated name
                        "enforce_boundaries", parameters=["self", "physics_obj"]
                    ),
                    MethodRequirement(
                        "update", parameters=["self", "physics_obj", "dt"]
                    ),
                    MethodRequirement(  # Added
                        "get_solar_irradiance", parameters=["self", "position"]
                    ),
                ],
            )
        ],
    ),
    "frame_core": ModuleRequirement(
        name="frame_core",
        component_type=ComponentType.HARDWARE,
        imports=IMPORT_REQUIREMENTS["frame_core"],
        classes=[
            ClassRequirement(
                name="StructuralSensors",
                methods=[
                    MethodRequirement("__init__", parameters=["self", "frame_core"]),
                    MethodRequirement(
                        "update",
                        parameters=[
                            "self",
                            "current_acceleration",
                            "current_angular_acceleration",
                        ],
                    ),
                    MethodRequirement("get_status", parameters=["self"]),
                ],
            ),
            ClassRequirement(
                name="FrameCore",
                methods=[
                    MethodRequirement("__init__", parameters=["self", "name"]),
                    MethodRequirement("get_total_mass", parameters=["self"]),
                    MethodRequirement(
                        "update",
                        parameters=[
                            "self",
                            "current_acceleration",
                            "current_angular_acceleration",
                        ],
                    ),
                    MethodRequirement("get_status", parameters=["self"]),
                ],
            ),
        ],
    ),
    "logger": ModuleRequirement(
        name="logger",
        component_type=ComponentType.UTILITY,
        imports=IMPORT_REQUIREMENTS["logger"],
        classes=[
            ClassRequirement(
                name="Logger",
                methods=[
                    MethodRequirement("__new__", parameters=["cls"]),
                    MethodRequirement("__init__", parameters=["self"]),
                    MethodRequirement(
                        "get_logger", parameters=["cls", "name"], is_classmethod=True
                    ),
                    MethodRequirement("info", parameters=["self", "message"]),
                    MethodRequirement("debug", parameters=["self", "message"]),
                    MethodRequirement("warning", parameters=["self", "message"]),
                    MethodRequirement("error", parameters=["self", "message"]),
                    MethodRequirement("critical", parameters=["self", "message"]),
                    MethodRequirement("set_level", parameters=["self", "level"]),
                    MethodRequirement("close", parameters=["self"]),
                    MethodRequirement("__del__", parameters=["self"]),
                ],
            )
        ],
    ),
    "physics": ModuleRequirement(
        name="physics",
        component_type=ComponentType.CORE,
        imports=IMPORT_REQUIREMENTS["physics"],
        classes=[
            ClassRequirement(
                name="State",
                methods=[
                    MethodRequirement(
                        "__init__",
                        parameters=[
                            "self",
                            "position",
                            "velocity",
                            "orientation",
                            "angular_velocity",
                        ],
                    ),  # __post_init__ убран
                    MethodRequirement("copy", parameters=["self"]),
                    MethodRequirement("__str__", parameters=["self"]),
                ],
            ),
            ClassRequirement(
                name="PhysicsObject",
                methods=[
                    MethodRequirement(
                        "__init__", parameters=["self", "name", "mass", "initial_state"]
                    ),
                    MethodRequirement("apply_force", parameters=["self", "force"]),
                    MethodRequirement("apply_torque", parameters=["self", "torque"]),
                    MethodRequirement("update", parameters=["self", "dt"]),
                    MethodRequirement("get_state", parameters=["self"]),
                    MethodRequirement(
                        "set_state", parameters=["self", "new_state"]
                    ),  # Добавлен set_state
                    MethodRequirement(
                        "get_forward_vector", parameters=["self"]
                    ),  # Добавлен get_forward_vector
                    # apply_impulse и get_orientation_matrix убраны, если не используются
                ],
            ),
        ],
    ),
    "power_systems": ModuleRequirement(
        name="power_systems",
        component_type=ComponentType.HARDWARE,
        imports=IMPORT_REQUIREMENTS["power_systems"],
        classes=[
            ClassRequirement(
                name="PowerSensors",
                methods=[
                    MethodRequirement("__init__", parameters=["self", "power_systems"]),
                    MethodRequirement(
                        "update",
                        parameters=[
                            "self",
                            "solar_power_generated_w",
                            "current_draw_w",
                        ],
                    ),
                    MethodRequirement(
                        "get_sensors_data", parameters=["self"]
                    ),  # Исправлено с get_status
                ],
            ),
            ClassRequirement(
                name="PowerSystems",
                methods=[
                    MethodRequirement("__init__", parameters=["self", "name"]),
                    MethodRequirement("get_total_mass", parameters=["self"]),
                    MethodRequirement("_calculate_power_demand", parameters=["self"]),
                    MethodRequirement(
                        "_generate_solar_power",
                        parameters=["self", "solar_irradiance_w_per_m2"],
                    ),
                    MethodRequirement(
                        "consume_power", parameters=["self", "power_w", "dt"]
                    ),
                    MethodRequirement(
                        "update", parameters=["self", "solar_irradiance_w_per_m2", "dt"]
                    ),
                    MethodRequirement("get_status", parameters=["self"]),
                ],
            ),
        ],
    ),
    "sensors": ModuleRequirement(
        name="sensors",
        component_type=ComponentType.SENSOR,
        imports=IMPORT_REQUIREMENTS["sensors"],
        classes=[
            ClassRequirement(
                name="Sensors",
                methods=[
                    MethodRequirement(
                        "__init__",
                        parameters=[
                            "self",
                            "physics_obj",
                            "frame_core",
                            "power_systems",
                            "thermal_system",
                        ],
                    ),
                    MethodRequirement(
                        "read_all", parameters=["self", "forces_applied"]
                    ),
                    MethodRequirement("_read_navigation", parameters=["self"]),
                    MethodRequirement(
                        "_read_frame_integrity", parameters=["self"]
                    ),  # Исправлено с _read_frame
                    MethodRequirement("_read_power", parameters=["self"]),
                    MethodRequirement("_read_thermal", parameters=["self"]),
                    MethodRequirement(
                        "check_critical_status", parameters=["self", "sensor_data"]
                    ),
                ],
            )
        ],
    ),
    "thermal_system": ModuleRequirement(
        name="thermal_system",
        component_type=ComponentType.HARDWARE,
        imports=IMPORT_REQUIREMENTS["thermal_system"],
        classes=[
            ClassRequirement(
                name="ThermalSystem",
                methods=[
                    MethodRequirement("__init__", parameters=["self", "name"]),
                    MethodRequirement(
                        "update", parameters=["self", "dt", "total_heat_generated_w"]
                    ),
                    MethodRequirement("_check_temperature_status", parameters=["self"]),
                    MethodRequirement("get_status", parameters=["self"]),
                ],
            )
        ],
    ),
    "visualizer": ModuleRequirement(
        name="visualizer",
        component_type=ComponentType.INTERFACE,
        imports=IMPORT_REQUIREMENTS["visualizer"],
        classes=[
            ClassRequirement(
                name="Visualizer",
                methods=[
                    MethodRequirement(
                        "__init__", parameters=["self"]
                    ),  # Убрал config, т.к. в коде его нет
                    MethodRequirement(
                        "_setup_dashboard", parameters=["self"]
                    ),  # Добавлен, т.к. используется
                    MethodRequirement(
                        "_update_dashboard",
                        parameters=[
                            "self",
                            "current_time",
                            "bot_state",
                            "sensor_data",
                            "session_steps",
                        ],
                    ),  # Добавлен
                    MethodRequirement(
                        "update",
                        parameters=[
                            "self",
                            "current_time",
                            "bot_state",
                            "sensor_data",
                            "session_steps",
                        ],
                    ),
                    MethodRequirement("close", parameters=["self"]),
                ],
            )
        ],
    ),
}

# Ожидаемое использование конфигурации
CONFIG_USAGE_REQUIREMENTS = {
    "main": {
        "config.BOT_MASS",
        "config.TIME_STEP",
        "config.MAX_TIME",
        "config.REAL_TIME_MODE",
    },
    "agent": {
        "config.MAX_VELOCITY",
        "config.LOW_BATTERY_THRESHOLD",
        "config.STRESS_DAMAGE_THRESHOLD",
        "config.WARNING_TEMPERATURE_C",
        "config.WARNING_TEMPERATURE_C_LOW",
        "config.FREEZE_THRESHOLD_C",
        "config.OVERHEAT_THRESHOLD_C",
        "config.CRITICAL_INTEGRITY_THRESHOLD",
        "config.TARGET_THRESHOLD",
        "config.THRUSTER_FORCE",
        "config.TORQUE_MAX",
        "config.BOUNDS",
        # MIN_CORE_TEMPERATURE_C и MAX_CORE_TEMPERATURE_C убраны, если агент их не использует напрямую
    },
    "analyzer": {
        "config.TIME_STEP",
    },
    "config": set(),  # Сам config.py не использует Config
    "environment": {
        "config.GRAVITY",
        "config.BOUNDS",
        "config.DEFAULT_SOLAR_IRRADIANCE_W_PER_M2",
    },
    "frame_core": {
        "config.FRAME_BASE_MASS",
        "config.MAX_ACCELERATION_TOLERANCE",
        "config.MAX_ANGULAR_ACCELERATION_TOLERANCE",
        "config.STRESS_DAMAGE_THRESHOLD",
    },
    "logger": set(),
    "physics": {
        # "config.GRAVITY", # Applied by environment module in main loop
    },
    "power_systems": {
        "config.MAIN_BATTERY_CAPACITY_WH",
        "config.RESERVE_BATTERY_CAPACITY_WH",
        "config.BATTERY_ENERGY_DENSITY_WH_PER_KG",
        "config.SOLAR_PANEL_1_MAX_W",
        "config.SOLAR_PANEL_2_MAX_W",
        "config.SOLAR_PANEL_3_MAX_W",
        "config.SOLAR_PANEL_POWER_DENSITY_W_PER_KG",
        "config.POWER_SYSTEM_BASE_MASS",
        "config.POWER_CONSUMPTION_IDLE_W",
        "config.POWER_CONSUMPTION_SENSORS_W",
        "config.POWER_CONSUMPTION_COMMUNICATION_IDLE_W",
        "config.POWER_CONSUMPTION_COMPUTATION_W",
        "config.POWER_CONSUMPTION_PROPULSION_ION_W",  # Используется
        "config.POWER_CONSUMPTION_COMMUNICATION_HIGH_W",  # Используется
        "config.SOLAR_PANEL_AREA_M2",
        "config.SOLAR_PANEL_EFFICIENCY",
        "config.DISCHARGE_EFFICIENCY",
        "config.MAX_CHARGE_RATE_W",
        # "config.MAX_DISCHARGE_RATE_W", # Not currently used
        "config.POWER_CONVERSION_EFFICIENCY",
        "config.OVERLOAD_THRESHOLD_W",
        "config.WARNING_BATTERY_THRESHOLD",
        "config.LOW_BATTERY_THRESHOLD",
        # Убраны неиспользуемые POWER_CONSUMPTION_*
    },
    "sensors": {
        # "config.MAIN_BATTERY_CAPACITY_WH", # Not directly used by sensors.py
        # "config.RESERVE_BATTERY_CAPACITY_WH", # Not directly used by sensors.py
        "config.LOW_BATTERY_THRESHOLD",
        "config.OVERHEAT_THRESHOLD_C",
        "config.CRITICAL_INTEGRITY_THRESHOLD",
        "config.STRESS_DAMAGE_THRESHOLD",
    },
    "thermal_system": {
        "config.INITIAL_CORE_TEMPERATURE_C",
        "config.INITIAL_RADIATOR_TEMPERATURE_C",
        "config.BOT_HEAT_CAPACITY_J_PER_C",
        "config.HEAT_TRANSFER_COEFFICIENT_W_PER_C",
        "config.AMBIENT_SPACE_TEMPERATURE_C",
        "config.MIN_CORE_TEMPERATURE_C",
        "config.MAX_CORE_TEMPERATURE_C",
        "config.OVERHEAT_THRESHOLD_C",
        "config.WARNING_TEMPERATURE_C",
        "config.FREEZE_THRESHOLD_C",
        "config.WARNING_TEMPERATURE_C_LOW",
    },
    "visualizer": {
        "config.PLOT_HISTORY",
        "config.UPDATE_RATE",
        "config.BOUNDS",
        "config.CRITICAL_INTEGRITY_THRESHOLD",
        "config.OVERHEAT_THRESHOLD_C",
        "config.WARNING_TEMPERATURE_C",
        "config.WARNING_TEMPERATURE_C_LOW",
        "config.TIME_STEP",  # Added for visualizer xlim adjustment
        # LOW_BATTERY_THRESHOLD и STRESS_DAMAGE_THRESHOLD убраны, если не используются для линий
    },
    "code_verifier": set(),  # code_verifier не должен использовать Config напрямую
}

# Обратная совместимость со старой схемой (если где-то используется)
EXPECTED_IMPORTS = {
    module_name: [imp.module for imp in req.imports]
    for module_name, req in MODULE_REQUIREMENTS.items()
}
EXPECTED_COMPONENTS = {
    module_name: {
        cls.name: [method.name for method in cls.methods] for cls in req.classes
    }
    for module_name, req in MODULE_REQUIREMENTS.items()
}
EXPECTED_CONFIG_USAGE = CONFIG_USAGE_REQUIREMENTS  # Уже в нужном формате

# Дополнительные валидации (можно расширить)
ARCHITECTURE_RULES = {
    "no_circular_imports": True,
    "max_cyclomatic_complexity": 15,  # Немного увеличил порог для некоторых модулей
    "min_docstring_coverage": 0.7,  # Немного снизил порог для начала
    "max_file_length": 700,
    "max_method_length": 70,
    "enforce_type_hints": True,
}

# Паттерны исключений для гибкости (можно расширить)
VALIDATION_EXCEPTIONS = {
    "main": {"allow_missing_docstrings": ["handle_signal"]},
    "config": {"allow_empty_methods": True, "min_docstring_coverage": 0.0},
    "logger": {"allow_long_parameter_lists": True, "max_cyclomatic_complexity": 20},
    "code_verifier": {
        "max_cyclomatic_complexity": 100,
        "min_docstring_coverage": 0.1,
        "max_file_length": 1000,
    },  # Исключения для самого верификатора
    "agent": {"max_cyclomatic_complexity": 25},
    "power_systems": {"max_cyclomatic_complexity": 20},
    "visualizer": {"max_method_length": 100},
}
