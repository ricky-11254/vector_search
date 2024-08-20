def two_sum(self: List[int], target: int) -> List[int]:
    d = {}
    for i, num in enumerate(self):
        if target - num in d:
            return [d[target - num], i]


