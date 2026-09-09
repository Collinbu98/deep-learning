from minigrad.value import Value


def test_addition():
    a = Value(2)
    b = Value(3)

    c = a + b

    assert c.data == 5


def test_multiplication():
    a = Value(2)
    b = Value(3)

    c = a * b

    assert c.data == 6


def test_computation_graph():
    x = Value(2)
    u = x * 3
    y = u * u

    assert u.data == 6
    assert y.data == 36
    assert x in u.parents
    assert u in y.parents
