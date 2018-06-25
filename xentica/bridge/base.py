"""
This module contains the main class to be used in custom bridges.

Methods from :class:`Bridge` class should be used in other bridges

"""

__all__ = ['Bridge', ]


class Bridge:
    """Main bridge class containing basic functions."""

    @staticmethod
    def noop(_model, _gui):
        """Do nothing."""

    @staticmethod
    def exit_app(_model, gui):
        """Exit GUI application."""
        gui.exit_app()

    @staticmethod
    def speed(dspeed):
        """Change simulation speed."""
        def func(model, _gui):
            """Apply speed."""
            model.apply_speed(dspeed)
        return func

    @staticmethod
    def toggle_pause(model, _gui):
        """Pause/unpause simulation."""
        model.toggle_pause()

    @staticmethod
    def toggle_sysinfo(_model, gui):
        """Turn system info panel on/off."""
        gui.sysinfo.toggle()
