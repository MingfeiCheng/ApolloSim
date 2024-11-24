import copy

from typing import List, Optional, Dict
from shapely.geometry import Polygon

from apollo_modules.modules.map.proto.map_crosswalk_pb2 import Crosswalk

class CrosswalkManager(object):

    def __init__(self, crosswalk: Optional[Dict[str, Crosswalk]] = None):
        self.crosswalk = crosswalk

    def setup(self, crosswalk: Dict):
        self.crosswalk = crosswalk

    def export(self):
        save_data = {
            'crosswalk': copy.deepcopy(self.crosswalk)
        }
        for k, y in save_data['crosswalk'].items():
            save_data['crosswalk'][k] = y.SerializeToString()
        return save_data

    def load(self, data: Dict):
        self.crosswalk = {}
        for k, y in data['crosswalk'].items():
            cw = Crosswalk()
            cw.ParseFromString(y)
            self.crosswalk[k] = cw

    def get_all(self) -> List[str]:
        return list(self.crosswalk.keys())


    def get(self, cw_id: str) -> Crosswalk:
        """
        Get a specific crosswalk object based on ID

        :param str cw_id: ID of the crosswalk interested in

        :returns: crosswalk object
        :rtype: Crosswalk
        """
        return self.crosswalk[cw_id]

    def get_polygon(self, cw_id: str) -> Polygon:
        cw = self.crosswalk[cw_id]
        cw_polygon_points = cw.polygon.point
        polygon_coords = []
        for point in cw_polygon_points:
            polygon_coords.append([point.x, point.y])
        start_point = polygon_coords[0]
        polygon_coords.append(start_point)
        cw_polygon = Polygon(polygon_coords)
        return cw_polygon
