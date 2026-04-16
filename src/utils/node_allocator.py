class NodeAllocator:
    def __init__(self, start: int = 1):
        self.counter = start

    def next_id(self) -> int:
        nid = self.counter
        self.counter += 1
        return nid
