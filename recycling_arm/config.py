SERIAL_PORT = "COM3"
BAUD_RATE = 9600
SERIAL_TIMEOUT = 2

BIN_ANGLES = {
    "plastic": (90, 70, 110, 40),
    "paper": (45, 70, 110, 40),
    "metal": (135, 70, 110, 40),
    "other": (90, 110, 70, 80),
}
HOME_ANGLES = (90, 90, 90, 90)

VALID_CATERGORY = tuple(BIN_ANGLES.keys())

def validate_config():
    if len(HOME_ANGLES) != 4:
        raise ValueError("HOME_ANGLES must contain four angles.")

    for catergory, angles in BIN_ANGLES.items():
        if len(angles) != 4:
            raise ValueError(f"{catergory!r} must have 4 servo angles.")
            
    for label, angles in [("HOME_ANGLES", HOME_ANGLES), *BIN_ANGLES.items()]:
        for angle in angles:
            if not isinstance(angle, (int, float)) or not (0 <= angle <= 180):
                raise ValueError(f"{label} contains an invalid angle: {angle!r}")

validate_config()