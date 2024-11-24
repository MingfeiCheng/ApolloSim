from apollo_sim.actor import register_agent, Location, BoundingBox
from apollo_sim.actor.static.base import StaticActor

@register_agent("static.traffic_cone")
class TrafficCone(StaticActor):

    # basic information - fixed
    category: str = 'static.traffic_cone'

    _bbox: BoundingBox = BoundingBox(
        length=0.5,
        width=0.5,
        height=1.0
    )

    def __init__(self, id: int, location: Location, role: str):
        super(TrafficCone, self).__init__(id, location, role)