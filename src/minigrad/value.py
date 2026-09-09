class Value:
    def __init__(self, data, parents=()):
        self.data = data
        self.grad = 0.0
        self.parents = parents
        self.backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other))

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out.backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other))

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out.backward = backward
        return out
