class Bridge:

    @staticmethod
    def noop(ca, gui):
        pass

    @staticmethod
    def exit_app(ca, gui):
        gui.exit_app()

    @staticmethod
    def speed(dspeed):
        def func(ca, gui):
            ca.apply_speed(dspeed)
        return func

    @staticmethod
    def toggle_pause(ca, gui):
        ca.toggle_pause()

    @staticmethod
    def toggle_sysinfo(ca, gui):
        gui.sysinfo.toggle()
