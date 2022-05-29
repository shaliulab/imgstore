# noinspection PyClassHasNoInit
class _Log:

    @staticmethod
    def info(msg):
        print(msg)

    debug = info
    warn = info
    error = info
