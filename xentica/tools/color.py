"""A collection of color conversion helpers."""


def hsv2rgb(hue, sat, val):
    """
    Convert HSV color to RGB format.

    :param hue: Hue value [0, 1]
    :param sat: Saturation value [0, 1]
    :param val: Brightness value [0, 1]

    :returns: tuple (red, green, blue)

    """
    raise NotImplementedError


def rgb2hsv(red, green, blue):
    """
    Convert RGB color to HSV format.

    :param red: Red value [0, 1]
    :param green: Green value [0, 1]
    :param blue: Blue value [0, 1]

    :returns: tuple (hue, sat, val)

    """
    raise NotImplementedError


def genome2rgb(genome):
    """
    Convert genome bit value to RGB color.

    :param genome: Genome as integer (bit) sequence.

    :returns: tuple (red, green, blue)

    """
    raise NotImplementedError
