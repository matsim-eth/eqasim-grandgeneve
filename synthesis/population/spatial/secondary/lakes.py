import numpy as np
import shapely.geometry as geo
import shapely.ops as ops
from shapely.prepared import prep

"""
Lakes as barriers for the secondary location assignment: the distance
distributions and the relaxation work with straight lines, so without this a
Thonon resident would go shopping in Lausanne, 13 km away across Lake Geneva
but about 55 km by land.

A trip is blocked if the shortest path around the lakes is more than
DETOUR_RATIO times its straight-line distance; short shore trips crossing a
bay (Thonon - Evian) or crossings of the tip of the lake in Geneva keep a
small detour and are not affected.
"""

DETOUR_RATIO = 1.5

# Lakes are simplified to compute the paths around them, and shrunk to test
# the crossings: the polygons are generalized, so shore locations can be a
# few hundred meters inside them.
SIMPLIFICATION = 500.0
SHRINKING = 200.0

class Lake:
    def __init__(self, polygon):
        polygon = polygon.simplify(SIMPLIFICATION)
        self.water = polygon.buffer(-SHRINKING)
        self.prepared_water = prep(self.water)
        self.polygon = polygon
        self.bounds = self.water.bounds

        rings = [polygon.exterior] if polygon.geom_type == "Polygon" else [part.exterior for part in polygon.geoms]
        self.vertices = np.array([coordinate for ring in rings for coordinate in ring.coords[:-1]])

        # Shortest paths between shore vertices (Floyd-Warshall on the
        # visibility graph)
        n = len(self.vertices)
        self.paths = np.full((n, n), np.inf)
        np.fill_diagonal(self.paths, 0.0)

        for i in range(n):
            for j in range(i + 1, n):
                if self.is_visible(self.vertices[i], self.vertices[j]):
                    self.paths[i, j] = self.paths[j, i] = np.linalg.norm(self.vertices[i] - self.vertices[j])

        for k in range(n):
            self.paths = np.minimum(self.paths, self.paths[:, k, np.newaxis] + self.paths[np.newaxis, k, :])

    def is_visible(self, a, b):
        return not self.prepared_water.intersects(geo.LineString([a, b]))

    def may_cross(self, a, b):
        xmin, ymin, xmax, ymax = self.bounds
        return not (
            max(a[0], b[0]) < xmin or min(a[0], b[0]) > xmax or
            max(a[1], b[1]) < ymin or min(a[1], b[1]) > ymax
        )

    def snap(self, point):
        """Moves points inside the lake to its shore."""
        shapely_point = geo.Point(point)

        if self.polygon.contains(shapely_point):
            return np.array(ops.nearest_points(self.polygon.exterior, shapely_point)[0].coords[0])

        return point

    def path_length(self, a, b):
        """Shortest path from a to b around the lake."""
        a, b = self.snap(a), self.snap(b)

        if self.is_visible(a, b):
            return np.linalg.norm(b - a)

        visible_a = [i for i, vertex in enumerate(self.vertices) if self.is_visible(a, vertex)]
        visible_b = [j for j, vertex in enumerate(self.vertices) if self.is_visible(b, vertex)]

        if len(visible_a) == 0 or len(visible_b) == 0:
            return np.inf

        distances_a = np.linalg.norm(self.vertices[visible_a] - a, axis = 1)
        distances_b = np.linalg.norm(self.vertices[visible_b] - b, axis = 1)

        return np.min(distances_a[:, np.newaxis] + self.paths[np.ix_(visible_a, visible_b)] + distances_b[np.newaxis, :])

class LakeBarrier:
    def __init__(self, polygons, detour_ratio = DETOUR_RATIO):
        self.detour_ratio = detour_ratio
        self.lakes = []

        for polygon in polygons:
            lake = Lake(polygon)

            if not lake.water.is_empty:
                self.lakes.append(lake)

    def is_blocked(self, a, b):
        """Whether the trip from a to b has to go around a lake with a
        detour above detour_ratio."""
        a, b = np.asarray(a, dtype = float).reshape(2), np.asarray(b, dtype = float).reshape(2)
        distance = np.linalg.norm(b - a)

        for lake in self.lakes:
            if lake.may_cross(a, b) and not lake.is_visible(a, b):
                if lake.path_length(a, b) > self.detour_ratio * distance:
                    return True

        return False

    def blocks_chain(self, problem, locations):
        """Whether one of the non-PT trips of the chain is blocked (PT trips
        may cross by boat)."""
        sequence = []
        if problem["origin"] is not None:
            sequence.append(problem["origin"])

        sequence += list(locations)

        if problem["destination"] is not None:
            sequence.append(problem["destination"])

        modes = problem["modes"]
        if len(modes) != len(sequence) - 1:
            modes = [None] * (len(sequence) - 1)

        for index, mode in enumerate(modes):
            if mode is not None and mode.startswith("pt"):
                continue

            if self.is_blocked(sequence[index], sequence[index + 1]):
                return True

        return False
