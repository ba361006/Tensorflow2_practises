class hello():
    def __init__(self):
        pass

    def __call__(self, input_value):
        print("hello, :{}".format(input_value))

a = hello()
a(1)