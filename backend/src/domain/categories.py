"""AudioSet category -> sensor mapping (shared between scripts and backend)."""

CATEGORY_SENSOR_MAP: dict[str, list[str]] = {
    "urban": [
        "Traffic noise, roadway noise",
        "Siren",
        "Car",
        "Emergency vehicle",
        "Engine",
        "Motor vehicle (road)",
        "Bus",
        "Truck",
        "Motorcycle",
        "Honking",
        "Jackhammer",
        "Sawing",
        "Filing (rasp)",
    ],
    "indoor": [
        "Air conditioning",
        "Mechanical fan",
        "Typing",
        "Keyboard (musical)",
        "Computer keyboard",
        "Footsteps",
        "Speech",
        "Conversation",
        "Door",
        "Knock",
        "Telephone",
        "Printer",
    ],
    "park": [
        "Bird",
        "Bird vocalization, bird call, bird song",
        "Chirp, tweet",
        "Wind",
        "Wind noise (microphone)",
        "Water",
        "Stream",
        "Rain",
        "Insect",
        "Cricket",
        "Rustling leaves",
        "Crow",
    ],
}


def build_label_to_sensor() -> dict[str, str]:
    """Build reverse mapping: label (lowercase) -> sensor_id."""
    mapping: dict[str, str] = {}
    for sensor_id, labels in CATEGORY_SENSOR_MAP.items():
        for label in labels:
            mapping[label.lower()] = sensor_id
    return mapping


LABEL_TO_SENSOR: dict[str, str] = build_label_to_sensor()
