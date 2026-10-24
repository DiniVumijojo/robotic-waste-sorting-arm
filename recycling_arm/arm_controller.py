import time
import serial

from config import (
    BAUD_RATE,
    BIN_ANGLES,
    HOME_ANGLES,
    SERIAL_PORT,
    SERIAL_TIMEOUT,
)

class ArmController:
    CATEGORIES = {"plastic", "paper", "metal", "other"}

    def __init__(self, port=SERIAL_PORT):
        self.connection = serial.Serial(
            port=port,
            baudrate=BAUD_RATE,
            timeout=SERIAL_TIMEOUT,
            write_timeout=SERIAL_TIMEOUT,
        )
        time.sleep(2)
        self.connection.reset_input_buffer()
        
        deadline = time.monotonic() + SERIAL_TIMEOUT
        while time.monotonic() < deadline:
            line = self.connection.readline().decode("ascii", errors="replace").strip()
            if line == "READY":
                break

    def send_command(self, command):
        command = command.strip().lower()
        allowed = {"home", "pick", "drop plastic", "drop paper", "drop metal", "drop other"}
        if command not in allowed:
            raise ValueError(f"Unsupported command: {command!r}")
            
        self.connection.write((command + "\n").encode("ascii"))
        self.connection.flush()
        
        response = self.connection.readline().decode("ascii", errors="replace").strip()
        if response != "OK":
            raise RuntimeError(f"Expected OK, got {response!r}")

    def sort_item(self, category):
        category = category.strip().lower()
        # FIXED: Corrected self.CATERGORIES to self.CATEGORIES
        if category not in self.CATEGORIES:
            category = "other"

        self.send_command("pick")
        self.send_command(f"drop {category}")
        self.send_command("home")

    def close(self):
        if getattr(self, "connection", None) and self.connection.is_open:
            self.connection.close()
            
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == "__main__":
    try:
        with ArmController() as arm:
            print(f"Connected to {SERIAL_PORT}. No movement command sent.")
    except serial.SerialException as error:
        print(f"Could not connect to the controller: {error}")