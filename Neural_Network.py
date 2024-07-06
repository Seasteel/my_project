import numpy
import scipy.special
import matplotlib.pyplot
import imageio
class NeuralNerwork:
    # Байгуулагч функц буву  утга олгох класса эхлүүлэх
     def __init__(self, inpputnodes, hiddennodes, outputnodes, learningrate):
         self.inodes = inpputnodes
         self.hnodes = hiddennodes
         self.onodes = outputnodes
         # w_i_j (неирон i-аас неирон j-рүү холбогдсон) w11 w21
         self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5),(self.hnodes, self.inodes))
         self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
         self.lr = learningrate
         self.activation_function = lambda x: scipy.special.expit(x)
         pass
     # Сургах функц
     def train(self, inputs_list, targets_list):
         inputs = numpy.array(inputs_list, ndmin=2).T
         targets = numpy.array(targets_list, ndmin=2).T
         # Сигналуудаа далд давхарга руу тооцоолох хэсэг
         hidden_inputs = numpy.dot(self.wih, inputs)
         # Идэвхжүүлэгч функц хэрэгжүүлэх хэсэг
         hidden_outputs = self.activation_function(hidden_inputs)

         #Сигналуудаа гаралтын давхарагат тооцолох хэсэг
         final_inputs = numpy.dot(self.who, hidden_outputs)
         #Сигналуудаа гаралтын давхарага дээр идэвхжүүлэгч функц дээр тооцоолох хэсэг
         final_outputs = self.activation_function(final_inputs)
         #Алдаа тооцоолох(target-actual)
         output_errors = targets - final_outputs
         #Далд давхарага дээрх алдаанууд
         hidden_errors = numpy.dot(self.who.T, output_errors)
         # Backprop ашиглан гаралтын давхрагаас далд давхрагаруу алдааг засах хэсэг
         self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
         # Backprop ашиглан далд давхрагаас оролтын давхрагаруу алдааг засах хэсэг
         self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *(1.0 - hidden_outputs)), numpy.transpose(inputs))
         pass
     # Сурсан хиймэл оюунаас асаалт асуух функц
     def query(self, input_list):
         # Оролтуудаа 2d массивруу хөрвүүлнэ
         inputs = numpy.array(input_list, ndmin=2).T
         # Сигналуудаа далд давхрага руу тооцоолох хэсэг
         hidden_inputs = numpy.dot(self.wih, inputs)
         hidden_outputs = self.activation_function(hidden_inputs)
         final_inputs = numpy.dot(self.who, hidden_outputs)
         final_outputs = self.activation_function(final_inputs)
         return final_outputs

#Оролтын, далд, гаралтын неиронууд
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
#Сургалтын хурд
learning_rate = 0.1
#неирон сүлжээгээ үүсгэх
n = NeuralNerwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

traindatafile = open('mnist.csv', 'r')
mnist = traindatafile.readlines()
traindatafile.close()
# Неирон сүлжээгээ сургах
epochs = 10
for e in range(epochs):
    #Сургалтын өгөгдлүүдээр давталт хийх
    for record in mnist:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    print('epoch: ', e)
    pass

img_array = imageio.imread('5.png', as_gray=True)
img_data = 255.0 - img_array.reshape(784)
img_data = (img_data / 255.0 * 0.99) + 0.01
print('min= ', numpy.min(img_data))
print('max= ', numpy.max(img_data))

matplotlib.pyplot.show(img_data.reshape(28, 28), cmap='Greys', interpolation='None')
outputs = n.query(img_data)
print('Асуусан: ', outputs)
label = numpy.argmax(outputs)
print(f'Таамаглаж байгаа тоо бол {label} юм.')