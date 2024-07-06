import numpy as np
import matplotlib.pyplot as plt

## 1.Өгөгдлөө бэлдэх хэсэг
x = np.random.normal(10, 70, 100)
y = np.random.normal(50, 40, 100)

## 2.Загвараа бэлдэх хэсэг
w1 = 0
w0 = 0
y_hat = w1*x+w0
## 3. Алдааны функц болон оновчлолын алгоритмаа тодорхойлно.
### 3.1 L = MSE, 3.2 Optimizer = GD
n = float(len(x)) ## Нийт сургах өгөгдлийн тоо
alpha = 0.0001 ## Сургалтын хурд
epoch = 1000 ## Сургах үе, итерацийн тоо

# Алдааг бууруулах машин сургалтын арга буюу Gradiant descent
for i in range(epoch):
    print(f'{i}-th epoch')
    y_hat = w1*x+w0
    L_w1 = (-2/n)*sum(x*(y-y_hat)) ##Алдааны функцээс w1 ээр уламжлал авсан функц
    L_w0 = (-2/n)*sum(y-y_hat)     ##Алдааны функцээс w0 ээр уламжлал авсан функц
    w1 = w1 - alpha*L_w1           ## w1 ийг засах томьёо
    w0 = w0 - alpha*L_w0           ## w0 ийг засах томьёо

#making predictions
y_hat = w1*x+w0
plt.scatter(x,y)
plt.plot([min(x),max(x)],[min(y_hat),max(y_hat)],color='red')
plt.show()
