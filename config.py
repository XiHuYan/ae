import numpy as np

class Config(object):
    # data directory root
    data_root = '/home/yanxuhua/data'

    # model configs
    middle_layer_size = [256, 128, 256]

    # regularized loss, (1-(x1^2+...+xn^2))^p
    p = 2

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
    


