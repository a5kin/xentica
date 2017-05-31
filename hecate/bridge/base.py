class Bridge:

    @staticmethod
    def noop(ca, gui):
        print("The action is not impelemented, passing by...")

    @staticmethod
    def exit_app(ca, gui):
        gui.exit_app()

    @staticmethod
    def move_up(ca, gui):
        ca.move(0, 1)

    @staticmethod
    def move_down(ca, gui):
        ca.move(0, -1)

    @staticmethod
    def move_left(ca, gui):
        ca.move(-1, 0)

    @staticmethod
    def move_right(ca, gui):
        ca.move(1, 0)

    @staticmethod
    def zoom_in(ca, gui):
        ca.apply_zoom(1)

    @staticmethod
    def zoom_out(ca, gui):
        ca.apply_zoom(-1)

    @staticmethod
    def speed_down(ca, gui):
        ca.apply_speed(-1)

    @staticmethod
    def speed_up(ca, gui):
        ca.apply_speed(1)

    @staticmethod
    def toggle_pause(ca, gui):
        ca.toggle_pause()
