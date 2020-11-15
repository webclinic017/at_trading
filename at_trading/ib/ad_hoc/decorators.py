# Python code to illustrate
# Decorators with parameters in Python

def decorator(*args, **kwargs):
    print("Inside decorator")

    def inner(func, *inner_args, **inner_kwargs):
        # code functionality here
        print("Inside inner function")
        print("I like", kwargs['like'])

        func(inner_args)

    # reurning inner function
    return inner


@decorator(like="geeksforgeeks")
def my_func(param1):
    print("Inside actual function {}".format(param1))
