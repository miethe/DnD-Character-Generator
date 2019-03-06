class Logger:
    def __init__(self, path, mode="a+", resume=False):
        self.path = path
        self.mode = mode
        self.resume = resume

        if self.resume is False:
            self._clean()

    def _clean(self):
        with open(self.path, "w") as f:
            pass

    def log(self, message):
        with open(self.path, self.mode) as f:
            f.write(message + "\n")
