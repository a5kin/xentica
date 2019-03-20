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
        hue_f = core.FloatVariable(hue * 6)
        hue_i = core.IntegerVariable(xmath.int(hue_f))
        hue_f -= hue_i
        sat = core.FloatVariable(sat)
        val = core.FloatVariable(val)
        grad_p = core.FloatVariable(val * (1 - sat))
        grad_q = core.FloatVariable(val * (1 - sat * hue_f))
        grad_t = core.FloatVariable(val * (1 - sat * (1 - hue_f)))
    else:
        hue_f = hue * 6
        hue_i = int(hue_f)
        hue_f -= hue_i
        grad_p = val * (1 - sat)
        grad_q = val * (1 - sat * hue_f)
        grad_t = val * (1 - sat * (1 - hue_f))

    red = (hue_i == 0 | hue_i >= 5) * val
    red = red + (hue_i == 1) * grad_q
    red = red + (hue_i == 2 | hue_i == 3) * grad_p
    red = red + (hue_i == 4) * grad_t

    green = (hue_i == 0 | hue_i >= 6) * grad_t
    green = green + (hue_i == 1 | hue_i == 2) * val
    green = green + (hue_i == 3) * grad_q
    green = green + (hue_i == 4 | hue_i == 5) * grad_p

    blue = (hue_i == 0 | hue_i == 1 | hue_i >= 6) * grad_p
    blue = blue + (hue_i == 2) * grad_t
    blue = blue + (hue_i == 3 | hue_i == 4) * val
    blue = blue + (hue_i == 5) * grad_q

    return (red, green, blue)


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
        d_red, d_green, d_blue = hsv2rgb(i / num_genes, 1, 1 / num_genes)
        val = (genome >> i) & 1
        red += d_red * val
        green += d_green * val
        blue += d_blue * val
    maxval = core.FloatVariable()
    maxval += xmath.max(red, green, blue)
    red /= maxval
    green /= maxval
    blue /= maxval
    return (red, green, blue)
