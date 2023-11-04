
duration_validator = lambda x: isinstance(x, str) and x.isdigit() and float(x) >= 0

test = '10. 3463'
print(duration_validator(test))
print(float(test))