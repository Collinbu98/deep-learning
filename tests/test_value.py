import pytest

from minigrad.value import Value, backward


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


def test_negation():
    a = Value(2)

    assert (-a).data == -2


def test_subtraction():
    a = Value(2)
    b = Value(5)

    assert (a - b).data == -3
    assert (a - 1).data == 1


def test_division():
    a = Value(3)
    b = Value(4)

    assert (a / b).data == 0.75
    assert (a / 2).data == 1.5


def test_tanh():
    a = Value(1)

    assert a.tanh().data == pytest.approx(0.7615941559557649)


def test_backward_accumulates_into_reused_parents():
    a = Value(3)

    # d(a * a + a * a) / da = 4a = 12
    y = a * a + a * a
    backward(y)

    assert y.grad == 1.0
    assert a.grad == pytest.approx(12.0)


def test_backward_uses_topological_order():
    a = Value(2)

    # y = (a + a) * (a - a); the node shared by both branches must be visited
    # before the multiplication that consumes its gradient.
    y = (a + a) * (a - a)
    backward(y)

    assert y.data == 0.0
    assert a.grad == 0.0


@pytest.mark.parametrize(
    "expression",
    [
        lambda a, b: a * b,
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b * a,
        lambda a, b: a.tanh() * b,
        lambda a, b: (a * b + a) / b,
    ],
)
def test_gradients_match_finite_differences(expression):
    a = Value(3.0)
    b = Value(5.0)
    backward(expression(a, b))

    eps = 1e-6

    def f(a_data, b_data):
        return expression(Value(a_data), Value(b_data)).data

    da = (f(3.0 + eps, 5.0) - f(3.0 - eps, 5.0)) / (2 * eps)
    db = (f(3.0, 5.0 + eps) - f(3.0, 5.0 - eps)) / (2 * eps)

    assert a.grad == pytest.approx(da)
    assert b.grad == pytest.approx(db)


def test_zero_grad():
    a = Value(3)

    y = a * a
    backward(y)
    a.zero_grad()

    assert a.grad == 0.0
