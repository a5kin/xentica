"""
This module contains the main class to be used in custom bridges.

Methods from :class:`Bridge` class should be used in other bridges

"""


class Bridge:
    """Main bridge class containing basic functions."""

    @staticmethod
    def noop(ca, gui):
        """Do nothing."""

    @staticmethod
    def exit_app(ca, gui):
        """Exit GUI application."""
        gui.exit_app()

    @staticmethod
    def speed(dspeed):
        """Change simulation speed."""
        def func(ca, gui):
            ca.apply_speed(dspeed)
        return func

    @staticmethod
    def toggle_pause(ca, gui):
        """Pause/unpause simulation."""
        ca.toggle_pause()

    @staticmethod
    def toggle_sysinfo(ca, gui):
        """Turn system info panel on/off."""
        gui.sysinfo.toggle()
