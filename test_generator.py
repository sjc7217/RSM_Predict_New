def gene():
    for i in range(10):
        yield i

a=gene()

print(a[10])
# for j in range(12):
#     print(next(a))