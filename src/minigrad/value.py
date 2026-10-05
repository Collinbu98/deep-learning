import math


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

    def __neg__(self):
        out = Value(-self.data, (self,))

        def backward():
            self.grad += -out.grad

        out.backward = backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data / other.data, (self, other))

        def backward():
            self.grad += out.grad / other.data
            other.grad += -out.grad * self.data / (other.data * other.data)

        out.backward = backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        return self + (-other)

    def tanh(self):
        t = math.tanh(self.data)

        out = Value(t, (self,))

        def backward():
            self.grad += (1 - t * t) * out.grad

        out.backward = backward
        return out

    def zero_grad(self):
        self.grad = 0.0


def backward(root):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for parent in v.parents:
                build_topo(parent)
            topo.append(v)

    build_topo(root)

    root.grad = 1.0

    for node in reversed(topo):
        node.backward()
