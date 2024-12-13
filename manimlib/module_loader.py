from __future__ import annotations

import builtins
import importlib
import os
import sys
import sysconfig

from manimlib.config import manim_config
from manimlib.logger import log

Module = importlib.util.types.ModuleType


class ModuleLoader:
    """
    Utility class to load a module from a file and handle its imports.

    Most parts of this class are only needed for the reload functionality,
    while the `get_module` method is the main entry point to import a module.
    """

    @staticmethod
    def get_module(file_name: str | None, is_during_reload=False) -> Module | None:
        """
        Imports a module from a file and returns it.

        During reload (when the user calls `reload()` in the IPython shell), we
        also track the imported modules and reload them as well (they would be
        cached otherwise). See the reload_manager where the reload parameter is set.

        Note that `exec_module()` is called twice when reloading a module:
        1. In exec_module_and_track_imports to track the imports
        2. Here to actually execute the module again with the respective
           imported modules reloaded.
        """
        if file_name is None:
            return None

        module_name = file_name.replace(os.sep, ".").replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, file_name)
        module = importlib.util.module_from_spec(spec)

        if is_during_reload:
            imported_modules = ModuleLoader._exec_module_and_track_imports(spec, module)
            reloaded_modules_tracker = set()
            ModuleLoader._reload_modules(imported_modules, reloaded_modules_tracker)

        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _exec_module_and_track_imports(spec, module: Module) -> set[str]:
        """
        Executes the given module (imports it) and returns all the modules that
        are imported during its execution.

        This is achieved by replacing the __import__ function with a custom one
        that tracks the imported modules. At the end, the original __import__
        built-in function is restored.
        """
        imported_modules: set[str] = set()
        original_import = builtins.__import__

        def tracked_import(name, globals=None, locals=None, fromlist=(), level=0):
            """
            Custom __import__ function that does exactly the same as the original
            one, but also tracks the imported modules by means of adding their
            names to a set.
            """
            result = original_import(name, globals, locals, fromlist, level)
            imported_modules.add(name)
            return result

        builtins.__import__ = tracked_import

        try:
            module_name = module.__name__
            log.debug('Reloading module "%s"', module_name)

            spec.loader.exec_module(module)
        finally:
            builtins.__import__ = original_import

        return imported_modules

    @staticmethod
    def _reload_modules(modules: set[str], reloaded_modules_tracker: set[str]):
        """
        Out of the given modules, reloads the ones that were not already imported.

        We skip modules that are not user-defined (see `is_user_defined_module()`).
        """
        for mod in modules:
            if mod in reloaded_modules_tracker:
                continue

            if not ModuleLoader._is_user_defined_module(mod):
                continue

            module = sys.modules[mod]
            ModuleLoader._deep_reload(module, reloaded_modules_tracker)

            reloaded_modules_tracker.add(mod)

    @staticmethod
    def _is_user_defined_module(mod: str) -> bool:
        """
        Returns whether the given module is user-defined or not.

        A module is considered user-defined if
        - it is not part of the standard library
        - AND it is not an external library (site-packages or dist-packages)
        """
        if mod not in sys.modules:
            return False

        if mod in sys.builtin_module_names:
            return False

        module = sys.modules[mod]
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            return False
        module_path = os.path.abspath(module_path)

        # External libraries (site-packages or dist-packages), e.g. numpy
        if "site-packages" in module_path or "dist-packages" in module_path:
            return False

        # Standard lib
        standard_lib_path = sysconfig.get_path("stdlib")
        if module_path.startswith(standard_lib_path):
            return False

        return True

    @staticmethod
    def _deep_reload(module: Module, reloaded_modules_tracker: set[str]):
        """
        Recursively reloads modules imported by the given module.

        Only user-defined modules are reloaded, see `is_user_defined_module()`.
        """
        ignore_manimlib_modules = manim_config.ignore_manimlib_modules_on_reload
        if ignore_manimlib_modules and module.__name__.startswith("manimlib"):
            return
        if module.__name__.startswith("manimlib.config"):
            # We don't want to reload global manim_config
            return

        if not hasattr(module, "__dict__"):
            return

        # Prevent reloading the same module multiple times
        if module.__name__ in reloaded_modules_tracker:
            return
        reloaded_modules_tracker.add(module.__name__)

        # Recurse for all imported modules
        for _attr_name, attr_value in module.__dict__.items():
            if isinstance(attr_value, Module):
                if ModuleLoader._is_user_defined_module(attr_value.__name__):
                    ModuleLoader._deep_reload(attr_value, reloaded_modules_tracker)

            # Also reload modules that are part of a class or function
            # e.g. when importing `from custom_module import CustomClass`
            elif hasattr(attr_value, "__module__"):
                attr_module_name = attr_value.__module__
                if ModuleLoader._is_user_defined_module(attr_module_name):
                    attr_module = sys.modules[attr_module_name]
                    ModuleLoader._deep_reload(attr_module, reloaded_modules_tracker)

        # Reload
        log.debug('Reloading module "%s"', module.__name__)
        importlib.reload(module)
