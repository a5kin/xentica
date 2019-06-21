"""The module with different helpers for CA rules."""
import re


class LifeLike:
    """Life-like rules helpers."""

    @staticmethod
    def golly2int(golly_str):
        """
        Convert a string in the Golly format to inner rule representation.

        :param golly_str: Rule in the Golly format (e.g. B3/S23)

        :returns: Integer representation of the rule for inner use.

        """
        born, sus = re.sub("[^0-9/]", "", golly_str).split("/", 1)
        born_rule = sum((1 << int(i)) for i in born)
        sustained_rule = sum((1 << (int(i) + 9)) for i in sus)
        return sustained_rule + born_rule

    @staticmethod
    def int2golly(rule):
        """
        Convert inner rule representation to string in the Golly format.

        :param rule: Integer representation of the rule.

        :returns: Golly-formatted rule (e.g. B3/S23).

        """
        sustained_rule = (rule >> 9) & 0b111111111
        born_rule = rule & 0b111111111
        sus = ''.join((str(i) * ((sustained_rule >> i) & 1)) for i in range(9))
        born = ''.join((str(i) * ((born_rule >> i) & 1)) for i in range(9))
        return "B%s/S%s" % (born, sus)
