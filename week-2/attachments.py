from typing import TypeVar, Self, Any

from decimal import Decimal, getcontext, ROUND_HALF_UP

import math

from functools import cached_property

context = getcontext()

Number = TypeVar("Number", int, float, Decimal)


class Point:
    """A class to represent a point in a 2D space.
    Attributes:
        x (Decimal): The x-coordinate of the point.
        y (Decimal): The y-coordinate of the point.
    Methods:
        delta_x(other: Point) -> Decimal:
            Calculate the difference in the x-coordinates between this point and another point.
        x_distance_to(other: Point) -> Decimal:
            Calculate the absolute difference in the x-coordinates between this point and another point.
        delta_y(other: Point) -> Decimal:
            Calculate the difference in the y-coordinates between this point and another point.
        y_distance_to(other: Point) -> Decimal:
            Calculate the absolute difference in the y-coordinates between this point and another point.
        distance_to(other: Point) -> Decimal:
            Calculate the distance from this point to another point.
        __eq__(other: Any) -> bool:
            Check if this point is equal to another point.
        __repr__() -> str:
            Return a string representation of the point."""

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


class Line:
    """A class to represent a line segment defined by two points.

    Attributes:
        point_1 (Point): The first point of the line segment.
        point_2 (Point): The second point of the line segment.
        slope (Decimal | None): The slope of the line segment.
        line_coefficients (tuple[Decimal, Decimal, Decimal]): The coefficients (A, B, C) of the line equation Ax + By = C.

    Methods:
        is_parallel_to(other: Self) -> bool:
            Check if the line segment is parallel to another line segment.
        is_perpendicular_to(other: Self) -> bool:
            Check if the line segment is perpendicular to another line segment.
        is_within_bounds(point: Point) -> bool:
            Check if a point is within the bounds of the line segment.
        contains_point(point: Point) -> bool:
            Check if the line segment contains a given point.
        contains_line(line: Self) -> bool:
            Check if the line segment contains another line segment.
        is_overlap(line: Self) -> bool:
            Check if the line segment overlaps with another line segment.
        intersection_point(other: Self) -> Point | None:
            Calculate the intersection point with another line segment, if it exists.
        has_intersection(other: Self) -> bool:
            Check if the line segment has an intersection with another line segment."""

    def __init__(self, point_1: Point, point_2: Point) -> None:
        if point_1 == point_2:
            raise ValueError("The two points of a line segment must be distinct.")

        self._point_1: Point = point_1
        self._point_2: Point = point_2

        self._slope: Decimal | None = None
        self._line_coefficients: tuple[Decimal, Decimal, Decimal] | None = None

    @property
    def point_1(self) -> Point:
        return self._point_1

    @point_1.setter
    def point_1(self, point: Point):
        self._line_coefficients = None
        self._slope = None

        self._point_1 = point

    @property
    def point_2(self) -> Point:
        return self._point_2

    @point_2.setter
    def point_2(self, point: Point):
        if self.point_1 == point:
            raise ValueError("The two points of a line segment must be distinct.")

        self._line_coefficients = None
        self._slope = None

        self._point_2 = point

    @cached_property
    def slope(self) -> Decimal | None:
        """Calculate the slope of the line defined by two points.

        If the line is vertical (delta_x is 0), the function returns None.

        Returns:
            Decimal | None: The slope of the line as a Decimal object, or None if the line is vertical.
        """

        if self._slope is None:
            a, b, _ = self.line_coefficients

            if b == 0:
                return None

            self._slope = -a / b

        return self._slope

    @cached_property
    def line_coefficients(self) -> tuple[Decimal, Decimal, Decimal]:
        """
        Calculate the coefficients of the line equation Ax + By = C
        that passes through two points.

        Returns:
            tuple[Decimal, Decimal, Decimal]: A tuple containing the coefficients
            (A, B, C) of the line equation.
        """
        if self._line_coefficients is None:
            a = self.point_2.delta_y(self.point_1)
            b = self.point_1.delta_x(self.point_2)
            c = a * self.point_1.x + b * self.point_1.y
            self._line_coefficients = a, b, c

        return self._line_coefficients

    def is_parallel_to(self, other: Self) -> bool:
        return self.slope == other.slope

    def is_perpendicular_to(self, other: Self) -> bool:
        if self.slope is None:
            return other.slope == 0
        elif self.slope == 0:
            return other.slope is None

        return False

    def is_within_bounds(self, point: Point) -> bool:
        """Check if a point is within the bounds of the line segment.

        Args:
            point (Point): The point to check.

        Returns:
            bool: True if the point is within the bounds, False otherwise.
        """

        return min(self.point_1.x, self.point_2.x) <= point.x <= max(
            self.point_1.x, self.point_2.x
        ) and min(self.point_1.y, self.point_2.y) <= point.y <= max(
            self.point_1.y, self.point_2.y
        )

    def contains_point(self, point: Point) -> bool:
        if point.x == self.point_1.x:
            return self.is_within_bounds(point) and self.slope is None
        else:
            return self.is_within_bounds(point) and self.slope == (
                point.y - self.point_1.y
            ) / (point.x - self.point_1.x)

    def contains_line(self, line: Self) -> bool:
        return (
            self.is_parallel_to(line)
            and self.is_within_bounds(line.point_1)
            and self.is_within_bounds(line.point_2)
        )

    def is_overlap(self, line: Self) -> bool:
        return self.is_parallel_to(line) and (
            self.is_within_bounds(line.point_1) or self.is_within_bounds(line.point_2)
        )

    def intersection_point(self, other: Self) -> Point | None:
        """Calculate the intersection point with another line, if it exists."""
        if self.is_parallel_to(other):
            return None

        a1, b1, c1 = self.line_coefficients
        a2, b2, c2 = other.line_coefficients

        determinant = a1 * b2 - a2 * b1
        if determinant == 0:
            return None

        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant

        intersection = Point(x, y)
        if self.is_within_bounds(intersection) and other.is_within_bounds(intersection):
            return intersection

        return None

    def has_intersection(self, other: Self) -> bool:
        return self.intersection_point(other) is not None

    def __repr__(self):
        return f"Line({self.point_1}, {self.point_2})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Line):
            return False

        return self.point_1 == other.point_1 and self.point_2 == other.point_2
