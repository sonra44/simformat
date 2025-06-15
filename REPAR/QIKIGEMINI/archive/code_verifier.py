"""
Эталонный верификатор кода проекта.
Обеспечивает комплексную проверку архитектуры, структуры и качества кода.
"""

import os
import ast
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter

try:
    from project_schema import (
        MODULE_REQUIREMENTS,
        CONFIG_USAGE_REQUIREMENTS,
        ARCHITECTURE_RULES,
        VALIDATION_EXCEPTIONS,
        ImportType,
        ComponentType,
        # Обратная совместимость
        EXPECTED_IMPORTS,
        EXPECTED_COMPONENTS,
        EXPECTED_CONFIG_USAGE,
    )
except ImportError:
    print("Warning: Could not import from project_schema. Using fallback mode.")
    EXPECTED_IMPORTS = {}
    EXPECTED_COMPONENTS = {}
    EXPECTED_CONFIG_USAGE = {}


class SeverityLevel(Enum):
    """Уровни серьезности проблем."""

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    STYLE = "STYLE"


@dataclass
class Issue:
    """Представление проблемы в коде."""

    file_path: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    severity: SeverityLevel = SeverityLevel.ERROR
    category: str = "General"
    code: str = "E001"
    message: str = ""
    suggestion: Optional[str] = None
    context: Optional[str] = None

    def __str__(self) -> str:
        location = f"{self.file_path}"
        if self.line_number:
            location += f":{self.line_number}"
            if self.column:
                location += f":{self.column}"

        result = f"[{self.severity.value}] {location} - {self.category}({self.code}): {self.message}"
        if self.suggestion:
            result += f"\n  Suggestion: {self.suggestion}"
        return result


@dataclass
class FileAnalysis:
    """Анализ отдельного файла."""

    file_path: str
    imports: Set[str] = field(default_factory=set)
    classes: Dict[str, List[str]] = field(default_factory=dict)
    functions: Set[str] = field(default_factory=set)
    config_usages: Set[str] = field(default_factory=set)
    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    docstring_coverage: float = 0.0
    issues: List[Issue] = field(default_factory=list)


class EtalonniyCodeVerifier:
    """
    Эталонный верификатор кода с расширенными возможностями анализа.
    """

    def __init__(self, project_root: str = ".", config_file: Optional[str] = None):
        self.project_root = Path(project_root).resolve()
        self.config = self._load_config(config_file)
        self.all_issues: List[Issue] = []
        self.file_analyses: Dict[str, FileAnalysis] = {}

        # Счетчики для статистики
        self.stats = {
            "files_processed": 0,
            "total_lines": 0,
            "issues_by_severity": Counter(),
            "issues_by_category": Counter(),
        }

        # Цветовая схема для консоли
        self.colors = {
            "CRITICAL": "\033[1;31m",  # Ярко-красный
            "ERROR": "\033[0;31m",  # Красный
            "WARNING": "\033[0;33m",  # Желтый
            "INFO": "\033[0;36m",  # Cyan
            "STYLE": "\033[0;35m",  # Пурпурный
            "SUCCESS": "\033[0;32m",  # Зеленый
            "RESET": "\033[0m",  # Сброс цвета
            "BOLD": "\033[1m",  # Жирный
        }

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Загружает конфигурацию верификатора."""
        default_config = {
            "ignore_patterns": [
                "__pycache__",
                ".git",
                ".pytest_cache",
                "venv",
                ".venv",
                "*.pyc",
            ],
            "max_line_length": 100,
            "enforce_docstrings": True,
            "check_type_hints": True,
            "check_complexity": True,
            "max_complexity": 10,
            "detailed_reports": True,
            "colorized_output": True,
        }

        if config_file and Path(config_file).exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self._print_warning(f"Could not load config file {config_file}: {e}")

        return default_config

    def _colorize(self, text: str, color_key: str) -> str:
        """Добавляет цветовое оформление к тексту."""
        if not self.config.get("colorized_output", True):
            return text
        return f"{self.colors.get(color_key, '')}{text}{self.colors['RESET']}"

    def _print_header(self, text: str):
        """Печатает заголовок с оформлением."""
        separator = "=" * len(text)
        print(self._colorize(f"\n{separator}", "BOLD"))
        print(self._colorize(text, "BOLD"))
        print(self._colorize(separator, "BOLD"))

    def _print_success(self, text: str):
        """Печатает успешное сообщение."""
        print(self._colorize(f"✓ {text}", "SUCCESS"))

    def _print_warning(self, text: str):
        """Печатает предупреждение."""
        print(self._colorize(f"⚠ {text}", "WARNING"))

    def _print_error(self, text: str):
        """Печатает ошибку."""
        print(self._colorize(f"✗ {text}", "ERROR"))

    def _get_module_name(self, filepath: Path) -> str:
        """Извлекает имя модуля из пути к файлу."""
        try:
            relative_path = filepath.relative_to(self.project_root)
            return str(relative_path.with_suffix("")).replace(os.sep, ".")
        except ValueError:
            return filepath.stem

    def _should_ignore_file(self, filepath: Path) -> bool:
        """Проверяет, следует ли игнорировать файл."""
        path_str = str(filepath)
        for pattern in self.config["ignore_patterns"]:
            if pattern.replace("*", "") in path_str:
                return True
        return False

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Вычисляет цикломатическую сложность узла AST."""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.ListComp):
                complexity += 1

        return complexity

    def _check_docstring_coverage(self, tree: ast.AST) -> Tuple[float, List[str]]:
        """Проверяет покрытие docstring в модуле."""
        total_items = 0
        documented_items = 0
        missing_docs = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if hasattr(node, "name") and not node.name.startswith("_"):
                    total_items += 1
                    if ast.get_docstring(node):
                        documented_items += 1
                    else:
                        missing_docs.append(f"{type(node).__name__}: {node.name}")

        coverage = documented_items / total_items if total_items > 0 else 1.0
        return coverage, missing_docs

    def _parse_python_file(self, filepath: Path) -> FileAnalysis:
        """Парсит Python файл и выполняет комплексный анализ."""
        analysis = FileAnalysis(file_path=str(filepath))

        try:
            with open(filepath, "r", encoding="utf-8-sig") as f:
                content = f.read()
                lines = content.split("\n")
                analysis.lines_of_code = len(
                    [
                        line
                        for line in lines
                        if line.strip() and not line.strip().startswith("#")
                    ]
                )

            tree = ast.parse(content, filename=str(filepath))

            # Анализ импортов
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis.imports.add(
                            alias.name
                        )  # Исправлено: используем полное имя модуля
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis.imports.add(node.module)

            # Анализ классов и функций
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)
                    analysis.classes[node.name] = methods
                elif isinstance(node, ast.FunctionDef):
                    analysis.functions.add(node.name)

            # Анализ использования Config
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id == "Config":
                        analysis.config_usages.add(f"config.{node.attr}")

            # Вычисление сложности
            analysis.cyclomatic_complexity = self._calculate_complexity(tree)

            # Проверка docstrings
            coverage, missing_docs = self._check_docstring_coverage(tree)
            analysis.docstring_coverage = coverage

            # Проверки качества кода
            self._check_code_quality(analysis, lines, tree)

        except SyntaxError as e:
            analysis.issues.append(
                Issue(
                    file_path=str(filepath),
                    line_number=e.lineno,
                    column=e.offset,
                    severity=SeverityLevel.CRITICAL,
                    category="Syntax",
                    code="C001",
                    message=f"Syntax error: {e.msg}",
                    context=str(e.text).strip() if e.text else None,
                )
            )
        except Exception as e:
            analysis.issues.append(
                Issue(
                    file_path=str(filepath),
                    severity=SeverityLevel.ERROR,
                    category="Parsing",
                    code="E001",
                    message=f"Failed to parse file: {str(e)}",
                    context=traceback.format_exc(),
                )
            )

        return analysis

    def _check_code_quality(
        self, analysis: FileAnalysis, lines: List[str], tree: ast.AST
    ):
        """Проверяет качество кода."""
        filepath = analysis.file_path

        # Проверка длины строк
        for i, line in enumerate(lines, 1):
            if len(line) > self.config["max_line_length"]:
                analysis.issues.append(
                    Issue(
                        file_path=filepath,
                        line_number=i,
                        severity=SeverityLevel.STYLE,
                        category="Style",
                        code="S001",
                        message=f"Line too long ({len(line)} > {self.config['max_line_length']})",
                        suggestion="Consider breaking this line into multiple lines",
                    )
                )

        # Проверка сложности
        if analysis.cyclomatic_complexity > self.config.get("max_complexity", 10):
            analysis.issues.append(
                Issue(
                    file_path=filepath,
                    severity=SeverityLevel.WARNING,
                    category="Complexity",
                    code="W002",
                    message=f"High cyclomatic complexity: {analysis.cyclomatic_complexity}",
                    suggestion="Consider refactoring into smaller functions",
                )
            )

        # Проверка docstrings
        if self.config.get("enforce_docstrings") and analysis.docstring_coverage < 0.8:
            analysis.issues.append(
                Issue(
                    file_path=filepath,
                    severity=SeverityLevel.WARNING,
                    category="Documentation",
                    code="W003",
                    message=f"Low docstring coverage: {analysis.docstring_coverage:.1%}",
                    suggestion="Add docstrings to classes and public functions",
                )
            )

    def _verify_imports(self, module_name: str, analysis: FileAnalysis) -> List[Issue]:
        """Верифицирует импорты модуля."""
        issues = []

        if module_name not in EXPECTED_IMPORTS:
            return issues

        expected_imports = set(EXPECTED_IMPORTS[module_name])
        actual_imports = analysis.imports

        # Проверка отсутствующих импортов
        missing_imports = expected_imports - actual_imports
        if missing_imports:
            issues.append(
                Issue(
                    file_path=analysis.file_path,
                    severity=SeverityLevel.ERROR,
                    category="Architecture",
                    code="E002",
                    message=f"Missing expected imports: {', '.join(sorted(missing_imports))}",
                    suggestion=f"Add imports: {', '.join(sorted(missing_imports))}",
                )
            )

        return issues

    def _verify_components(
        self, module_name: str, analysis: FileAnalysis
    ) -> List[Issue]:
        """Верифицирует компоненты модуля."""
        issues = []

        if module_name not in EXPECTED_COMPONENTS:
            return issues

        expected_components = EXPECTED_COMPONENTS[module_name]

        for class_name, expected_methods in expected_components.items():
            if class_name not in analysis.classes:
                issues.append(
                    Issue(
                        file_path=analysis.file_path,
                        severity=SeverityLevel.ERROR,
                        category="Architecture",
                        code="E003",
                        message=f"Missing expected class: '{class_name}'",
                        suggestion=f"Implement class {class_name} with methods: {', '.join(expected_methods)}",
                    )
                )
            else:
                actual_methods = set(analysis.classes[class_name])
                missing_methods = set(expected_methods) - actual_methods
                if missing_methods:
                    issues.append(
                        Issue(
                            file_path=analysis.file_path,
                            severity=SeverityLevel.ERROR,
                            category="Architecture",
                            code="E004",
                            message=f"Class '{class_name}' missing methods: {', '.join(sorted(missing_methods))}",
                            suggestion=f"Add missing methods to {class_name} class",
                        )
                    )

        return issues

    def _verify_config_usage(
        self, module_name: str, analysis: FileAnalysis
    ) -> List[Issue]:
        """Верифицирует использование конфигурации."""
        issues = []

        if module_name not in EXPECTED_CONFIG_USAGE:
            return issues

        expected_config = EXPECTED_CONFIG_USAGE[module_name]
        actual_config = analysis.config_usages

        # Отладочная печать
        # print(f"DEBUG: Verifying config for {module_name}")
        # print(f"DEBUG: expected_config type: {type(expected_config)}, value: {expected_config}")
        # print(f"DEBUG: actual_config type: {type(actual_config)}, value: {actual_config}")

        missing_config = expected_config - actual_config
        if missing_config:
            issues.append(
                Issue(
                    file_path=analysis.file_path,
                    severity=SeverityLevel.WARNING,
                    category="Configuration",
                    code="W004",
                    message=f"Missing expected Config usage: {', '.join(sorted(missing_config))}",
                    suggestion="Ensure all required configuration parameters are used",
                )
            )

        unexpected_config = actual_config - expected_config # Эта операция тоже set - set
        if unexpected_config:
            issues.append(
                Issue(
                    file_path=analysis.file_path,
                    severity=SeverityLevel.INFO,
                    category="Configuration",
                    code="I001",
                    message=f"Unexpected Config usage: {', '.join(sorted(unexpected_config))}",
                    suggestion="Verify if these configuration parameters are necessary",
                )
            )

        return issues

    def verify_project(self) -> bool:
        """Верифицирует весь проект."""
        self._print_header("🔍 ЭТАЛОННАЯ ВЕРИФИКАЦИЯ ПРОЕКТА")

        self.all_issues.clear()
        self.file_analyses.clear()
        self.stats = {
            "files_processed": 0,
            "total_lines": 0,
            "issues_by_severity": Counter(),
            "issues_by_category": Counter(),
        }

        # Поиск и анализ Python файлов
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Исключаем игнорируемые директории
            dirs[:] = [
                d
                for d in dirs
                if not any(
                    pattern.replace("*", "") in d
                    for pattern in self.config["ignore_patterns"]
                )
            ]

            for file in files:
                if file.endswith(".py") and file != "project_schema.py":
                    filepath = Path(root) / file
                    if not self._should_ignore_file(filepath):
                        python_files.append(filepath)

        if not python_files:
            self._print_warning("No Python files found for verification")
            return True

        print(f"Found {len(python_files)} Python files to analyze...")

        # Анализ каждого файла
        for filepath in python_files:
            try:
                print(f"Analyzing: {filepath.relative_to(self.project_root)}")

                analysis = self._parse_python_file(filepath)
                module_name = self._get_module_name(filepath).split(".")[-1]

                # Базовые проверки структуры
                structure_issues = []
                structure_issues.extend(self._verify_imports(module_name, analysis))
                structure_issues.extend(self._verify_components(module_name, analysis))
                structure_issues.extend(
                    self._verify_config_usage(module_name, analysis)
                )

                # Объединяем все проблемы
                all_file_issues = analysis.issues + structure_issues

                # Сохраняем результаты
                analysis.issues = all_file_issues
                self.file_analyses[str(filepath)] = analysis
                self.all_issues.extend(all_file_issues)

                # Обновляем статистику
                self.stats["files_processed"] += 1
                self.stats["total_lines"] += analysis.lines_of_code

                for issue in all_file_issues:
                    self.stats["issues_by_severity"][issue.severity.value] += 1
                    self.stats["issues_by_category"][issue.category] += 1

            except Exception as e:
                error_issue = Issue(
                    file_path=str(filepath),
                    severity=SeverityLevel.CRITICAL,
                    category="System",
                    code="C002",
                    message=f"Failed to analyze file: {str(e)}",
                    context=traceback.format_exc(),
                )
                self.all_issues.append(error_issue)
                self.stats["issues_by_severity"]["CRITICAL"] += 1

        return (
            len(
                [
                    issue
                    for issue in self.all_issues
                    if issue.severity in [SeverityLevel.CRITICAL, SeverityLevel.ERROR]
                ]
            )
            == 0
        )

    def print_detailed_report(self):
        """Выводит детальный отчет о проверке."""
        self._print_header("📊 ДЕТАЛЬНЫЙ ОТЧЕТ АНАЛИЗА")

        # Общая статистика
        print(f"\n{self._colorize('Общая статистика:', 'BOLD')}")
        print(f"  Файлов обработано: {self.stats['files_processed']}")
        print(f"  Всего строк кода: {self.stats['total_lines']}")
        print(f"  Всего проблем: {len(self.all_issues)}")

        # Статистика по уровням серьезности
        print(f"\n{self._colorize('Проблемы по уровням:', 'BOLD')}")
        for severity in ["CRITICAL", "ERROR", "WARNING", "INFO", "STYLE"]:
            count = self.stats["issues_by_severity"][severity]
            if count > 0:
                color_key = severity
                print(f"  {self._colorize(severity, color_key)}: {count}")

        # Статистика по категориям
        if self.stats["issues_by_category"]:
            print(f"\n{self._colorize('Проблемы по категориям:', 'BOLD')}")
            for category, count in self.stats["issues_by_category"].most_common():
                print(f"  {category}: {count}")

        # Детальный список проблем
        if self.all_issues:
            print(f"\n{self._colorize('ДЕТАЛЬНЫЙ СПИСОК ПРОБЛЕМ:', 'BOLD')}")
            print("-" * 80)

            # Группируем по файлам
            issues_by_file = defaultdict(list)
            for issue in self.all_issues:
                issues_by_file[issue.file_path].append(issue)

            for file_path in sorted(issues_by_file.keys()):
                print(f"\n{self._colorize(f'📄 {file_path}', 'BOLD')}")

                # Сортируем проблемы по серьезности и номеру строки
                severity_order = {
                    SeverityLevel.CRITICAL: 0,
                    SeverityLevel.ERROR: 1,
                    SeverityLevel.WARNING: 2,
                    SeverityLevel.INFO: 3,
                    SeverityLevel.STYLE: 4,
                }

                file_issues = sorted(
                    issues_by_file[file_path],
                    key=lambda x: (severity_order[x.severity], x.line_number or 0),
                )

                for issue in file_issues:
                    color_key = issue.severity.value
                    print(f"  {self._colorize('●', color_key)} {issue}")

        # Рекомендации по улучшению
        self._print_improvement_suggestions()

    def _print_improvement_suggestions(self):
        """Выводит рекомендации по улучшению кода."""
        print(f"\n{self._colorize('РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:', 'BOLD')}")

        critical_count = self.stats["issues_by_severity"]["CRITICAL"]
        error_count = self.stats["issues_by_severity"]["ERROR"]
        warning_count = self.stats["issues_by_severity"]["WARNING"]

        if critical_count > 0:
            print(
                f"  🔥 {self._colorize('КРИТИЧНО', 'CRITICAL')}: Немедленно исправьте {critical_count} критических проблем"
            )

        if error_count > 0:
            print(
                f"  ❌ {self._colorize('ВЫСОКИЙ ПРИОРИТЕТ', 'ERROR')}: Исправьте {error_count} ошибок архитектуры"
            )

        if warning_count > 0:
            print(
                f"  ⚠️  {self._colorize('СРЕДНИЙ ПРИОРИТЕТ', 'WARNING')}: Рассмотрите {warning_count} предупреждений"
            )

        # Специфичные рекомендации
        if self.stats["issues_by_category"].get("Style", 0) > 10:
            print(
                "  📝 Рекомендуется настроить автоформатирование кода (black, autopep8)"
            )

        if self.stats["issues_by_category"].get("Documentation", 0) > 5:
            print("  📚 Улучшите документирование кода - добавьте docstrings")

        if self.stats["issues_by_category"].get("Complexity", 0) > 3:
            print("  🔄 Рефакторинг: упростите сложные функции")

    def print_summary_report(self) -> int:
        """Выводит краткий отчет и возвращает код выхода."""
        self._print_header("✨ ИТОГОВЫЙ ОТЧЕТ")

        total_issues = len(self.all_issues)
        critical_issues = self.stats["issues_by_severity"]["CRITICAL"]
        error_issues = self.stats["issues_by_severity"]["ERROR"]

        if total_issues == 0:
            self._print_success(
                "Отлично! Проблем не найдено. Код соответствует эталону."
            )
            self._print_success(
                f"Проанализировано {self.stats['files_processed']} файлов"
            )
            print("\n🏆 " + self._colorize("СТАТУС: ЭТАЛОННОЕ КАЧЕСТВО", "SUCCESS"))
            return 0

        print(f"\nВсего найдено проблем: {total_issues}")

        if critical_issues > 0:
            self._print_error(f"Критических проблем: {critical_issues}")
            print("\n💥 " + self._colorize("СТАТУС: КРИТИЧЕСКИЕ ОШИБКИ", "CRITICAL"))
            return 2

        if error_issues > 0:
            self._print_error(f"Ошибок: {error_issues}")
            print("\n🚨 " + self._colorize("СТАТУС: ТРЕБУЮТСЯ ИСПРАВЛЕНИЯ", "ERROR"))
            return 1

        self._print_warning(
            f"Только предупреждения и стилевые замечания: {total_issues}"
        )
        print("\n⚠️  " + self._colorize("СТАТУС: ХОРОШО, ЕСТЬ ЗАМЕЧАНИЯ", "WARNING"))
        return 0

    def save_report(self, filename: str = "code_verification_report.json"):
        """Сохраняет отчет в JSON файл."""
        report_data = {
            "timestamp": (
                str(pd.Timestamp.now()) if "pd" in globals() else str(datetime.now())
            ),
            "project_root": str(self.project_root),
            "statistics": dict(self.stats),
            "issues": [
                {
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "column": issue.column,
                    "severity": issue.severity.value,
                    "category": issue.category,
                    "code": issue.code,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                    "context": issue.context,
                }
                for issue in self.all_issues
            ],
        }

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            self._print_success(f"Отчет сохранен в {filename}")
        except Exception as e:
            self._print_error(f"Не удалось сохранить отчет: {e}")


def main():
    """Главная функция для запуска верификатора."""
    import argparse

    parser = argparse.ArgumentParser(description="Эталонный верификатор кода проекта")
    parser.add_argument("--root", "-r", default=".", help="Корневая директория проекта")
    parser.add_argument("--config", "-c", help="Путь к файлу конфигурации")
    parser.add_argument("--detailed", "-d", action="store_true", help="Детальный отчет")
    parser.add_argument("--save", "-s", help="Сохранить отчет в файл")
    parser.add_argument(
        "--no-color", action="store_true", help="Отключить цветной вывод"
    )

    args = parser.parse_args()

    # Создаем верификатор
    verifier = EtalonniyCodeVerifier(project_root=args.root, config_file=args.config)

    if args.no_color:
        verifier.config["colorized_output"] = False

    try:
        # Запускаем верификацию
        success = verifier.verify_project()

        # Выводим отчеты
        if args.detailed:
            verifier.print_detailed_report()

        exit_code = verifier.print_summary_report()

        # Сохраняем отчет если нужно
        if args.save:
            verifier.save_report(args.save)

        return exit_code

    except KeyboardInterrupt:
        print("\n\n⏹️  Верификация прервана пользователем")
        return 130
    except Exception as e:
        print(f"\n💥 Критическая ошибка верификатора: {e}")
        if args.detailed:
            traceback.print_exc()
        return 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
