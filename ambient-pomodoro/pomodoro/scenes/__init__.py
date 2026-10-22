"""Scene registry.

To add a scene: create a module in this package with a :class:`Scene`
subclass (give it a unique ``key`` and a ``title``), then add the class to
``SCENES`` below. Order here is the order the V key cycles through.
"""

from __future__ import annotations

from .aurora import AuroraScene
from .base import FieldScene, Scene
from .gradient import GradientScene
from .lava_lamp import LavaLampScene

SCENES: list[type[Scene]] = [
    LavaLampScene,
    AuroraScene,
    GradientScene,
]

DEFAULT_SCENE = SCENES[0].key


def scene_keys() -> list[str]:
    return [cls.key for cls in SCENES]


def get_scene_class(key: str) -> type[Scene]:
    """Look up a scene by key, falling back to the first registered scene."""
    for cls in SCENES:
        if cls.key == key:
            return cls
    return SCENES[0]


def next_scene_key(key: str, step: int = 1) -> str:
    keys = scene_keys()
    try:
        index = keys.index(key)
    except ValueError:
        return keys[0]
    return keys[(index + step) % len(keys)]


__all__ = [
    "SCENES",
    "DEFAULT_SCENE",
    "FieldScene",
    "Scene",
    "get_scene_class",
    "next_scene_key",
    "scene_keys",
]
