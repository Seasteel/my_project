def checkPerfectNumber(n):
    s = 0
    for i in range(1, n):
        if n % i == 0:
            s += i
    if s == n:
        print(f'{n} is perfect number')
    else:
        print(f'{n} is perfect number')


def isPerfect(n):
    for i in range(1, n + 1):
        sum = 0
        for j in range(1, i):
            if i % j == 0:
                sum += j
        if i == sum:
            print(i)

isPerfect(1000)

