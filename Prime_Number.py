n = int(input('Тоо оруул: '))
if n < 1:
    print('Not prime')
else:
    for i in range(2, n):
        if n % i == 0:
            print('Not Prime')# 2 3 4 5 6 7 8 9
            break
    else:
        print('Prime Number')