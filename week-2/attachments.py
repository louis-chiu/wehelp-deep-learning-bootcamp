from typing import TypeVar, Self, Any

from decimal import Decimal, getcontext, ROUND_HALF_UP

context = getcontext()

Number = TypeVar("Number", int, float)


class Point:
    def __init__(self, x: Number, y: Number) -> None:
        self._x: Decimal = Decimal(f"{x}")
        self._y: Decimal = Decimal(f"{y}")

    @property
    def x(self) -> Decimal:
        return self._x

    @x.setter
    def x(self, x: Number):
        self._x = Decimal(f"{x}")

    @property
    def y(self) -> Decimal:
        return self._y

    @y.setter
    def y(self, y: Number):
        self._y = Decimal(f"{y}")

    def delta_x(self, other: Self) -> Decimal:
        return self.x - other.x

    def x_distance_to(self, other: Self) -> Decimal:
        return abs(self.delta_x(other))

    def delta_y(self, other: Self) -> Decimal:
        return self.y - other.y

    def y_distance_to(self, other: Self) -> Decimal:
        return abs(self.delta_y(other))

    def distance_to(self, other: Self) -> Decimal:
        """Calculate the distance from this point to another point.

        Args:
            other (Point): The other point to which the distance is calculated.

        Returns:
            Decimal: The distance between the two points, rounded to three decimal places.

        Raises:
            TypeError: If 'other' is not an instance of Point.
        """

        if not isinstance(other, Point):
            raise TypeError(
                f"Expected 'other' to be a Point, but got {type(other).__name__}."
            )

        return (
            (self.x_distance_to(other) ** 2 + self.y_distance_to(other) ** 2)
            .sqrt(context)
            .quantize(Decimal(".000"), ROUND_HALF_UP)
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"
