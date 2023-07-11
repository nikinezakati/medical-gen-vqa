import datetime
from src.constants import LOGS_DIR
import os
class Logger:
    
    def __init__(self, logs_dir):
        self.logs_dir = logs_dir

    def log(self, module, message):
        os.makedirs(self.logs_dir, exist_ok=True)
        logs_path = f"{self.logs_dir}/{module}.logs"
        message = f"[{datetime.datetime.now()}] {message}" + "\n"
        method = 'a' if os.path.exists(logs_path) else 'w'
        with open(logs_path, method) as f:
                f.write(message)

Instance  = Logger(LOGS_DIR)