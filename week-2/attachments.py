from typing import TypeVar, Self, Any

from decimal import Decimal
from math import hypot

Number = TypeVar("Number", int, float)


class Point:
    def __init__(self, x: Number, y: Number) -> None:
        self._x: Decimal = Decimal("{x}")
        self._y: Decimal = Decimal("{y}")

    @property
    def x(self) -> Decimal:
        return self._x

    @x.setter
    def x(self, x: Number):
        self._x = Decimal("{x}")

    @property
    def y(self) -> Decimal:
        return self._y

    @y.setter
    def y(self, y: Number):
        self._y = Decimal("{y}")

    def delta_x(self, other: Self) -> Decimal:
        return self.x - other.x

    def x_distance_to(self, other: Self) -> Decimal:
        return abs(self.delta_x(other))

    def delta_y(self, other: Self) -> Decimal:
        return self.y - other.y

    def y_distance_to(self, other: Self) -> Decimal:
        return abs(self.delta_y(other))

    def distance_to(self, other: Self) -> Decimal:
        if not isinstance(other, Point):
            raise TypeError(
                f"Expected 'other' to be a Point, but got {type(other).__name__}."
            )

        return Decimal(hypot(self.x_distance_to(other), self.y_distance_to(other)))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"
