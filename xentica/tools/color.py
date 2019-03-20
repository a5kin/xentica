"""A collection of color conversion helpers."""

from xentica import core
from xentica.core.expressions import DeferredExpression
from xentica.tools import xmath


def hsv2rgb(hue, sat, val):
    """
    Convert HSV color to RGB format.

    :param hue: Hue value [0, 1]
    :param sat: Saturation value [0, 1]
    :param val: Brightness value [0, 1]

    :returns: tuple (red, green, blue)

    """
    if isinstance(hue, DeferredExpression):
        f = core.FloatVariable(hue * 6)
        hi = core.IntegerVariable(xmath.int(f))
        f -= hi
        s = core.FloatVariable(sat)
        v = core.FloatVariable(val)
        p = core.FloatVariable(v * (1 - s))
        q = core.FloatVariable(v * (1 - s * f))
        t = core.FloatVariable(v * (1 - s * (1 - f)))
    else:
        f = hue * 6
        hi = int(f)
        f -= hi
        s = sat
        v = val
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

    r = (hi == 0 | hi >= 5) * v
    r = r + (hi == 1) * q
    r = r + (hi == 2 | hi == 3) * p
    r = r + (hi == 4) * t

    g = (hi == 0 | hi >= 6) * t
    g = g + (hi == 1 | hi == 2) * v
    g = g + (hi == 3) * q
    g = g + (hi == 4 | hi == 5) * p

    b = (hi == 0 | hi == 1 | hi >= 6) * p
    b = b + (hi == 2) * t
    b = b + (hi == 3 | hi == 4) * v
    b = b + (hi == 5) * q

    return (r, g, b)


def genome2rgb(genome, num_genes):
    """
    Convert genome bit value to RGB color.

    :param genome:
        Genome as integer (bit) sequence.
    :param num_genes:
        Genome length in bits.

    :returns: tuple (red, green, blue)

    """
    red = core.FloatVariable()
    green = core.FloatVariable()
    blue = core.FloatVariable()
    for i in range(num_genes):
        dr, dg, db = hsv2rgb(i / num_genes, 1, 1 / num_genes)
        val = (genome >> i) & 1
        red += dr * val
        green += dg * val
        blue += db * val
    maxval = core.FloatVariable()
    maxval += xmath.max(red, green, blue)
    red /= maxval
    green /= maxval
    blue /= maxval
    return (red, green, blue)
