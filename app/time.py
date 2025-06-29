import time


class Timer:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.checkpoint = time.time()

    def print(self) -> None:
        print(f"Execution time: {time.time() - self.checkpoint:.4f} seconds\n")
