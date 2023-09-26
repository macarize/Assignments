import random
from matplotlib import pyplot as plt

X = [[3, 4, 5], [3, 2, 6]]
y = [2, 7, 3]
k = [1, 4, 2]
w = [0, 0]

class MyModel():
    def __init__(self, x, y, w, k = None):
        if k == None:
            self.weight = []
            for i in range(len(y)) :
                self.weight.append(random.random())
        else :
            self.weight = k
        self.w = w

    def pred(self, x):
        pred = []
        shape_of_x = (len(x), len(x[0]))

        weights = self.w
        for i in range(shape_of_x[0]) :
            elem = 0
            for j in range(shape_of_x[1]) :
                elem += weights[j] * x[i][j]
            pred.append(elem)
        return pred

    def Weighted_MSE(self, pred, y):
        loss_accumulated = 0
        diff = []
        for i in range(len(pred)):
            diff.append(pred[i] - y[i])
        for i in range(len(diff)) :
            diff[i] = diff[i] * diff[i]
            loss_accumulated += self.weight[i] * diff[i]
        loss = loss_accumulated / len(pred)
        return loss
    # def SDG(self, x, y, lr=0.005):
    def SGD(self, x, ym, lr):
        random_sample_idx = random.randrange(0, len(self.weight))
        random_sample = [x[random_sample_idx]]

        grad_1 = self.weight[random_sample_idx] * ((self.pred(random_sample)[0] - y[random_sample_idx])* random_sample[0][0])
        grad_2 = self.weight[random_sample_idx] * ((self.pred(random_sample)[0] - y[random_sample_idx])* random_sample[0][1])

        self.w[0] = self.w[0] - lr * grad_1
        self.w[1] = self.w[1] - lr * grad_2
        return [self.w[0], self.w[1]]
lr = 0.005
X = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]
obj = MyModel(X, y, w, k)

loss_array = []
weight_array = []
for i in range(100):
    pred = obj.pred(X)
    loss = obj.Weighted_MSE(obj.pred(X), y)
    loss_array.append(loss)
    weight = obj.SGD(X, y, lr)
    weight_array.append(weight)
    if i == 4:
        print("weight of t = 5 : {}".format(obj.w))
    if i == 9:
        print("weight of t = 10 : {}".format(obj.w))
    if i == 49:
        print("weight of t = 50 : {}".format(obj.w))
    if i == 99:
        print("weight of t = 100 : {}".format(obj.w))
print("Prediction : {}".format(obj.pred(X)))
print("Loss : {}".format(obj.Weighted_MSE(obj.pred(X), y)))
print("Final weight : {}".format(weight_array[len(weight_array)-1]))

'''Loss plot'''
plt.plot(range(1, 101), loss_array)
plt.show()

'''weight plot'''
weight_array_T = [[weight_array[j][i] for j in range(len(weight_array))] for i in range(len(weight_array[0]))]
plt.plot(range(1, 101), weight_array_T[0])
plt.plot(range(1, 101), weight_array_T[1])
plt.show()
