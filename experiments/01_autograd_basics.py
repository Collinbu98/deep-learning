from minigrad.value import Value


x = Value(2)

u = x * 3

y = u * u

print(f"x = {x.data}")
print(f"u = {u.data}")
print(f"y = {y.data}")

print()
print("y parents:", [parent.data for parent in y.parents])
print("u parents:", [parent.data for parent in u.parents])
