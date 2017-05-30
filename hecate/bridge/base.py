class Bridge:

    @staticmethod
    def noop(ca, gui):
        print("The action is not impelemented, passing by...")

    @staticmethod
    def exit_app(ca, gui):
        gui.exit_app()
