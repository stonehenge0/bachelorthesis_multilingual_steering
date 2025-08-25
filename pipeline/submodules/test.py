def print_something(text: str) -> None:
    "Prints a string."
    print(text)
    return text.upper()


def return_doubled(value: int) -> int:
    "Return double of a given int"
    return value * 2


def test(text: str, value: int) -> tuple[str, int]:
    """Run a test"""
    upper_text = print_something(text)
    doubled_value = return_doubled(value)
    return upper_text, doubled_value
