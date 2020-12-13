def generator():
    a = True
    for i in range(10):
        print('generator')
        if a:
            yield i
        else:
            yield i*2
        
def gg():
    print('hi')
    for i in generator():
        yield i+10

for i in gg():
    print(i)