class Bridge:

    @staticmethod
    def noop(ca, gui):
        print("The action is not impelemented, passing by...")

    @staticmethod
    def exit_app(ca, gui):
        gui.exit_app()

    @staticmethod
    def scroll_up(ca, gui):
        ca.scroll(0, 1)

    @staticmethod
    def scroll_down(ca, gui):
        ca.scroll(0, -1)

    @staticmethod
    def scroll_left(ca, gui):
        ca.scroll(-1, 0)

    @staticmethod
    def scroll_right(ca, gui):
        ca.scroll(1, 0)

    @staticmethod
    def zoom_in(ca, gui):
        ca.zoomed(1)

    @staticmethod
    def zoom_out(ca, gui):
        ca.zoomed(-1)
