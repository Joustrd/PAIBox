from pathlib import Path
from typing import Union

from paibox.context import _Context
from paibox.libpaicore import Coord, CoordLike, to_coord

__all__ = ["BACKEND_CONFIG"]

DEFAULT_OUTPUT_CHIP_ADDR = Coord(1, 0)
DEFAULT_LOCAL_CHIP_ADDR = Coord(0, 0)
DEFAULT_OUTPUT_CORE_ADDR_START = Coord(0, 0)


class _BackendContext(_Context):
    def __init__(self) -> None:
        super().__init__()
        self._context["output_chip_addr"] = DEFAULT_OUTPUT_CHIP_ADDR
        self._context["local_chip_addr"] = DEFAULT_LOCAL_CHIP_ADDR
        self._context["build_directory"] = Path.cwd()
        self._context["output_core_addr_start"] = DEFAULT_OUTPUT_CORE_ADDR_START

    @property
    def local_chip_addr(self) -> Coord:
        return self["local_chip_addr"]

    @local_chip_addr.setter
    def local_chip_addr(self, addr: CoordLike) -> None:
        self["local_chip_addr"] = to_coord(addr)

    @property
    def test_chip_addr(self) -> Coord:
        return self["output_chip_addr"]

    @test_chip_addr.setter
    def test_chip_addr(self, addr: CoordLike) -> None:
        self["output_chip_addr"] = to_coord(addr)

    @property
    def output_dir(self) -> Path:
        return self["build_directory"]

    @output_dir.setter
    def output_dir(self, p: Union[str, Path]) -> None:
        self["build_directory"] = Path(p)


_BACKEND_CONTEXT = _BackendContext()
BACKEND_CONFIG = _BACKEND_CONTEXT
