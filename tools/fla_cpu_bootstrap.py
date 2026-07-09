"""
fla CPU-only bootstrap for macOS.

短路 fla.ops + triton 的整条 import 链:
  - triton 无 macOS wheel
  - fla.ops 依赖 triton 的 GPU kernel
  - 转换脚本只做权重字典搬运、不跑 forward,不需要任何 ops/triton 实现

策略:
  1. meta path finder 拦截 "fla.ops.*",返回万能 stub package,短路所有 ops。
  2. triton 用一个"半智能" stub module:
     - 顶层符号 (jit/autotune/Config/__version__) 提供合法实现,让
       `inspect.signature(triton.autotune)` 等通过;
     - `triton.runtime.driver` 访问抛 AttributeError,让 fla 的
       `get_available_device()` 走 except 回退到 'cpu'。

仅在"纯 CPU 权重转换"场景使用,绝不可用于训练/推理。
"""
import sys
import types
import inspect
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec


# ── 通用万能值 ──────────────────────────────────────────────
class _Sentinel:
    """可调用、可迭代、可作为任意属性值,且 inspect.signature 友好。"""
    __slots__ = ()

    def __call__(self, *a, **k): return _SENTINEL
    def __getattr__(self, n): return _SENTINEL
    def __getitem__(self, k): return _SENTINEL
    def __iter__(self): return iter(())
    def __bool__(self): return False

    # 类型注解场景: `X | None` 需要 __or__/__ror__
    def __or__(self, other): return _SENTINEL
    def __ror__(self, other): return _SENTINEL

    # 作为基类场景: class Foo(StubSentinel) 需 __mro_entries__ 返回 tuple
    def __mro_entries__(self, bases): return ()

    @property
    def __signature__(self):
        return inspect.Signature()

_SENTINEL = _Sentinel()


# ── fla.ops 万能 stub ───────────────────────────────────────
class _StubModule(types.ModuleType):
    """万能 stub 模块: 任意属性/调用都返回 _SENTINEL。"""

    def __init__(self, name):
        super().__init__(name)
        self.__name__ = name
        self.__path__ = []
        self.__package__ = name
        self.__file__ = None
        self.__spec__ = ModuleSpec(name, _STUB_LOADER, is_package=True)
        self.__loader__ = _STUB_LOADER

    def __getattr__(self, name):
        return _SENTINEL

    def __call__(self, *a, **k):
        return _SENTINEL


class _StubLoader(Loader):
    def create_module(self, spec):
        return _StubModule(spec.name)
    def exec_module(self, module):
        return None

_STUB_LOADER = _StubLoader()


# ── triton 半智能 stub ──────────────────────────────────────
def _stub_jit(*a, **k):
    """模仿 triton.jit: 给函数注入 arg_names,返回原函数。"""
    if a and callable(a[0]) and not k:
        fn = a[0]
    else:
        def deco_inner(fn): return fn
        return deco_inner
    try:
        sig = inspect.signature(fn)
        fn.arg_names = list(sig.parameters.keys())
    except (ValueError, TypeError):
        fn.arg_names = []
    return fn


class _TritonRuntimeModule(types.ModuleType):
    """triton.runtime: driver 访问抛异常,触发 fla 回退到 CPU。

    fla.utils._device.get_available_device() 访问链:
        triton.runtime.driver.active.get_current_target().backend
    让 driver 抛 AttributeError,即可走 except → 'cpu'。
    autotuner 子模块需要被 import (Autotuner/autotune),提供 stub。
    """
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []
        self.__package__ = name
        self.__file__ = None
        self.__spec__ = ModuleSpec(name, _STUB_LOADER, is_package=True)
        self.__loader__ = _STUB_LOADER

    @property
    def driver(self):
        raise AttributeError("triton driver unavailable on CPU stub")

    def __getattr__(self, name):
        return _SENTINEL


class _TritonClassModule(types.ModuleType):
    """triton 子模块但返回真实空 class (非 _Sentinel)。

    torch 的 `from triton.runtime.jit import JITFunction` 随后会
    `isinstance(x, JITFunction)`,JITFunction 必须是真实 type。
    fla 的 `from triton.runtime.autotuner import Autotuner` 类似。
    """
    def __init__(self, name, classes=()):
        super().__init__(name)
        self.__path__ = []
        self.__package__ = name
        self.__file__ = None
        self.__spec__ = ModuleSpec(name, _STUB_LOADER, is_package=True)
        self.__loader__ = _STUB_LOADER
        for cls_name in classes:
            object.__setattr__(self, cls_name, type(cls_name, (), {}))


class _TritonModule(types.ModuleType):
    """triton 顶层: 提供合法的 jit/autotune/Config/__version__。"""

    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []
        self.__package__ = name
        self.__file__ = None
        self.__spec__ = ModuleSpec(name, _STUB_LOADER, is_package=True)
        self.__loader__ = _STUB_LOADER
        object.__setattr__(self, "__version__", "3.3.0")
        object.__setattr__(self, "jit", _stub_jit)
        object.__setattr__(self, "autotune", _stub_jit)
        object.__setattr__(self, "heuristics", _stub_jit)
        object.__setattr__(self, "Config", lambda *a, **k: None)
        object.__setattr__(self, "constexpr", lambda *a, **k: (lambda f: f))
        object.__setattr__(self, "language", _StubModule(name + ".language"))

    # 注意: 不提供 __getattr__ 兜底,否则会拦截子模块 import (如 triton.runtime)
    # 子模块由 meta path finder / sys.modules 负责。


# ── meta path finder ───────────────────────────────────────
# 需要"返回真实空 class"的 triton 子模块及其类名。
# torch: from triton.runtime.jit import JITFunction
# fla:   from triton.runtime.autotuner import Autotuner, autotune
_TRITON_CLASS_MODULES = {
    "triton.runtime.jit": ("JITFunction", "_JITFunction"),
    "triton.runtime.autotuner": ("Autotuner", "AutotunerWrapper"),
}


class _Finder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "fla.ops" or fullname.startswith("fla.ops."):
            return ModuleSpec(fullname, _STUB_LOADER, is_package=True)
        if fullname == "triton":
            return _TritonSpec(fullname)
        if fullname == "triton.runtime":
            return _RuntimeSpec(fullname)
        if fullname in _TRITON_CLASS_MODULES:
            return _ClassModuleSpec(fullname, _TRITON_CLASS_MODULES[fullname])
        if fullname.startswith("triton."):
            return ModuleSpec(fullname, _STUB_LOADER, is_package=True)
        return None


class _TritonSpec(ModuleSpec):
    def __init__(self, name):
        super().__init__(name, _TritonLoader(), is_package=True)


class _RuntimeSpec(ModuleSpec):
    def __init__(self, name):
        super().__init__(name, _RuntimeLoader(), is_package=True)


class _ClassModuleSpec(ModuleSpec):
    def __init__(self, name, classes):
        super().__init__(name, _ClassModuleLoader(classes), is_package=True)


class _TritonLoader(Loader):
    def create_module(self, spec):
        return _TritonModule(spec.name)
    def exec_module(self, module):
        return None


class _RuntimeLoader(Loader):
    def create_module(self, spec):
        return _TritonRuntimeModule(spec.name)
    def exec_module(self, module):
        return None


class _ClassModuleLoader(Loader):
    def __init__(self, classes):
        self._classes = classes
    def create_module(self, spec):
        return _TritonClassModule(spec.name, self._classes)
    def exec_module(self, module):
        return None


def install():
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    for top in ("fla.ops", "triton"):
        if top not in sys.modules:
            if top == "triton":
                sys.modules[top] = _TritonModule(top)
            else:
                sys.modules[top] = _StubModule(top)
    if "triton.runtime" not in sys.modules:
        sys.modules["triton.runtime"] = _TritonRuntimeModule("triton.runtime")
    for modname, classes in _TRITON_CLASS_MODULES.items():
        if modname not in sys.modules:
            sys.modules[modname] = _TritonClassModule(modname, classes)


if __name__ == "__main__":
    install()
    import fla  # noqa
    from fla.models.rwkv7 import RWKV7Config, RWKV7ForCausalLM  # noqa
    print("fla import OK under CPU bootstrap")
