import pkgutil
import boxmot
import inspect

print(f"boxmot version: {boxmot.__version__ if hasattr(boxmot, '__version__') else 'unknown'}")
print(f"boxmot location: {boxmot.__file__}")
print("\nTop-level boxmot attributes:")
print([n for n in dir(boxmot) if not n.startswith("_")])

print("\nSearching all submodules for tracker classes...")
for _, modname, _ in pkgutil.walk_packages(boxmot.__path__, prefix="boxmot."):
    try:
        mod = __import__(modname, fromlist=["dummy"])
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ == modname and ("track" in name.lower() or "sort" in name.lower()):
                print(f"  {modname}.{name}")
    except Exception:
        pass
