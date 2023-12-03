

class ClusterHead:
    def __init__(
        self,
        id: int,
        position: list[float], #(x,y,z)
        speed: float,
        max_range: float,
        current_range: float,
        energy: float,
    ) -> None:
        self.id = id
        self.position = position
        self.speed = speed
        self.max_range = max_range
        self.current_range = current_range
        self.energy = energy
    
    def updatePosition(
        self,
        location: list[float]
    ) -> float:
        if self.position == location:
            return 0
        else:
            distance = np.linalg.norm(location - self.position)
            if distance < self.speed:
                self.position = location
                return distance
            else:
                ratio = self.speed/distance
                self.position = [
                    (1-ratio)*self.position[0] + ratio*location[0],
                    (1-ratio)*self.position[1] + ratio*location[1],
                    (1-ratio)*self.position[2] + ratio*location[2]]
                return self.speed

