import os

class EnvContextManager:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.old_value = None

    def __enter__(self):
        self.old_value = os.environ.get(self.key)
        os.environ[self.key] = self.value

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.old_value is not None:
            os.environ[self.key] = self.value
        else:
            del os.environ[self.key]

            assert self.key not in os.environ