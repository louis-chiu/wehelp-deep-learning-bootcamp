from typing import TypeVar, Self, Any

from decimal import Decimal, getcontext, ROUND_HALF_UP
from abc import ABC
import math

from itertools import combinations

context = getcontext()

Number = TypeVar("Number", int, float, Decimal)


# Task 1
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

    def __add__(self, other: Any) -> Self:
        if not isinstance(other, Vector):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'Point' and '{type(other).__name__}'"
            )
        self.x += other.x_offset
        self.y += other.y_offset
        return self

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

    @property
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

    @property
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
        if self.slope is None and other.slope is None:
            return True
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

    def length(self) -> Decimal:
        return self.point_1.distance_to(self.point_2)

    def __repr__(self):
        return f"Line({self.point_1}, {self.point_2})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Line):
            return False

        return self.point_1 == other.point_1 and self.point_2 == other.point_2


class Circle:
    """A class to represent a circle in a 2D space.

    Attributes:
        center (Point): The center point of the circle.
        radius (Decimal): The radius of the circle.
        area (Decimal): The area of the circle.

    Methods:
        has_intersection(other: Circle) -> bool:
            Check if the circle has an intersection with another circle.
    """

    def __init__(self, center: Point, radius: Number):
        if radius == 0:
            raise ValueError("The radius of a circle cannot be zero.")

        self._center: Point = center
        self._radius: Decimal = Decimal(f"{radius}")

        self._area = None

    @property
    def center(self) -> Point:
        return self._center

    @center.setter
    def center(self, center: Point):
        self._center = center

    @property
    def radius(self) -> Decimal:
        return self._radius

    @radius.setter
    def radius(self, radius: Number):
        self._area = None
        self._radius = Decimal(f"{radius}")

    @property
    def area(self) -> Decimal:
        if self._area is None:
            self._area = (Decimal(f"{(math.pi)}") * self.radius**2).quantize(
                Decimal(".000"), ROUND_HALF_UP
            )

        return self._area

    def has_intersection(self, other: Self) -> bool:
        return self.center.distance_to(other.center) < (self.radius + other.radius)

    def contains(self, other: Point) -> bool:
        return self.center.distance_to(other) <= self.radius


class Polygon:
    def __init__(self, point_1: Point, point_2: Point, point_3: Point, point_4: Point):
        if not self.__can_form_quadrilateral(point_1, point_2, point_3, point_4):
            raise ValueError("The four points cannot form a quadrilateral.")

        self._point_1: Point = point_1
        self._point_2: Point = point_2
        self._point_3: Point = point_3
        self._point_4: Point = point_4

    def __can_form_quadrilateral(
        self, point_1: Point, point_2: Point, point_3: Point, point_4: Point
    ) -> bool:
        """Check if four points can form a quadrilateral."""

        points = [point_1, point_2, point_3, point_4]
        for point1, point2 in combinations(points, 2):
            if point1 == point2:
                return False

        lines = [
            Line(point_1, point_2),
            Line(point_2, point_3),
            Line(point_3, point_4),
            Line(point_4, point_1),
        ]

        for line1, line2 in combinations(lines, 2):
            if line1.is_parallel_to(line2) and line1.is_overlap(line2):
                return False

        return True

    @property
    def point_1(self) -> Point:
        return self._point_1

    @property
    def point_2(self) -> Point:
        return self._point_2

    @property
    def point_3(self) -> Point:
        return self._point_3

    @property
    def point_4(self) -> Point:
        return self._point_4

    @property
    def perimeter(self) -> Decimal:
        return (
            Line(self.point_1, self.point_2).length()
            + Line(self.point_2, self.point_3).length()
            + Line(self.point_3, self.point_4).length()
            + Line(self.point_4, self.point_1).length()
        )

    def __repr__(self) -> str:
        return (
            f"Polygon({self.point_1}, {self.point_2}, {self.point_3}, {self.point_4})"
        )


def task_1():
    COLOR_CODE: str = "\033[31m"
    COLOR_RESET_CODE: str = "\033[0m"

    line_a = Line(Point(-6, 1), Point(2, 4))
    line_b = Line(Point(-6, -1), Point(2, 2))
    line_c = Line(Point(-4, -4), Point(-1, 6))
    print(
        f"Are Line A and Line B parallel? {COLOR_CODE}{line_a.is_parallel_to(line_b)}{COLOR_RESET_CODE}"
    )
    print(
        f"Are Line C and Line A perpendicular? {COLOR_CODE}{line_c.is_perpendicular_to(line_a)}{COLOR_RESET_CODE}"
    )

    circle_a = Circle(Point(6, 3), 2)
    circle_b = Circle(Point(8, 1), 1)
    print(f"Print the area of Circle A. {COLOR_CODE}{circle_a.area}{COLOR_RESET_CODE}")
    print(
        f"Do Circle A and Circle B intersect? {COLOR_CODE}{circle_a.has_intersection(circle_b)}{COLOR_RESET_CODE}"
    )

    polygon_a = Polygon(
        Point(-1, -2),
        Point(2, 0),
        Point(5, -1),
        Point(4, -4),
    )
    print(
        f"Print the perimeter of Polygon A. {COLOR_CODE}{polygon_a.perimeter}{COLOR_RESET_CODE}"
    )


# Task 2
class Vector:
    def __init__(self, x_offset: Number, y_offset: Number):
        self._x_offset: Decimal = Decimal(f"{x_offset}")
        self._y_offset: Decimal = Decimal(f"{y_offset}")

    @property
    def x_offset(self) -> Decimal:
        return self._x_offset

    @x_offset.setter
    def x_offset(self, x_offset: Number):
        self._x_offset = Decimal(f"{x_offset}")

    @property
    def y_offset(self) -> Decimal:
        return self._y_offset

    @y_offset.setter
    def y_offset(self, y_offset: Number):
        self._y_offset = Decimal(f"{y_offset}")

    def __add__(self, other: Any) -> Self | Point:
        if isinstance(other, Vector):
            self.x_offset += other.x_offset
            self.y_offset += other.y_offset
            return self
        elif isinstance(other, Point):
            other.x += self.x_offset
            other.y += self.y_offset
            return other
        raise TypeError(
            f"Unsupported operand type(s) for +: 'Point' and '{type(other).__name__}'"
        )

    def __mul__(self, other: Any) -> Self:
        if not isinstance(other, int):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'Point' and '{type(other).__name__}'"
            )
        self.x_offset *= other
        self.y_offset *= other
        return self


class Location(Point):
    def __init__(self, x: int, y: int):
        super().__init__(x, y)

    def __repr__(self) -> str:
        return f"({self.x:>3}, {self.y:>3})"


class Enemy:
    def __init__(self, name: str, location: Location, vector: Vector):
        self._name: str = name
        self._location: Location = location
        self._vector: Vector = vector

        self._points: int = 10

    @property
    def name(self) -> str:
        return self._name

    @property
    def points(self) -> int:
        return self._points

    @points.setter
    def points(self, points: int):
        self._points = points

    @property
    def location(self) -> Location:
        return self._location

    @location.setter
    def location(self, location: Location):
        self._location = location

    @property
    def vector(self) -> Decimal:
        return self._vector

    def is_alive(self) -> bool:
        return self.points > 0

    def move(self, times: int = 1) -> Location:
        self.location = self.location + self.vector * times
        return self.location


class DefenseRange(Circle):
    def __init__(self, location: Location, range: int):
        super().__init__(location, range)

    @property
    def location(self):
        return super().center

    @property
    def range(self):
        return super().radius


class Tower(ABC):
    def __init__(self, name: str, power: int, defense_range: DefenseRange):
        self._name: str = name
        self._power: int = power
        self._defense_range: DefenseRange = defense_range

    @property
    def name(self) -> str:
        return self._name

    @property
    def power(self) -> int:
        return self._power

    @property
    def defense_range(self) -> Circle:
        return self._defense_range

    def search_alive_enemies_within_range(self, enemies: list[Enemy]) -> list[Enemy]:
        return [
            enemy
            for enemy in enemies
            if self.defense_range.contains(enemy.location) and enemy.is_alive()
        ]

    def attack_all(self, enemies: list[Enemy]):
        for enemy in enemies:
            enemy.points = enemy.points - self.power


class BasicTower(Tower):
    def __init__(self, name: str, location: Location):
        super().__init__(name, 1, DefenseRange(location, 2))


class AdvancedTower(Tower):
    def __init__(self, name: str, location: Location):
        super().__init__(name, 2, DefenseRange(location, 4))


def task_2():
    print(f"=========={' Task2 Game Start ':^18}==========")

    enemies = [
        Enemy("E1", Location(-10, 2), Vector(2, -1)),
        Enemy("E2", Location(-8, 0), Vector(3, 1)),
        Enemy("E3", Location(-9, -1), Vector(3, 0)),
    ]

    towers = [
        BasicTower("T1", Location(-3, 2)),
        BasicTower("T2", Location(-1, -2)),
        BasicTower("T3", Location(4, 2)),
        BasicTower("T4", Location(7, 0)),
        AdvancedTower("A1", Location(1, 1)),
        AdvancedTower("A2", Location(4, -3)),
    ]

    for _ in range(10):
        alive_enemies = [enemy for enemy in enemies if enemy.is_alive()]
        for alive_enemy in alive_enemies:
            alive_enemy.move()

        for tower in towers:
            detected_enemies = tower.search_alive_enemies_within_range(alive_enemies)
            if len(detected_enemies) != 0:
                tower.attack_all(detected_enemies)

    print(
        "\n".join(
            [
                "Enemy {}, HP {} at {}".format(enemy.name, enemy.points, enemy.location)
                for enemy in enemies
            ]
        )
    )
    print(f"=========={' Task2 Game Over ':^18}==========")


def main():
    task_1()
    print()
    task_2()


if __name__ == "__main__":
    main()
