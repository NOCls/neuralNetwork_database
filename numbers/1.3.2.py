import numpy 

def sigmoid(x):
    return 1.0/(1+numpy.exp(-x))

def sigmoid_derivative(x):
    return x*(1.0-x)

class NeuralNetwork:
    def __init__(self,x,y):
        self.input =x #4_row 3_line   as(4,3)
        self.weights1 =numpy.random.rand(self.input.shape[1],4)  #self.input.shape[1]
        self.weights2 =numpy.random.rand(4,1) 
        
        self.y =y
        self.output =numpy.zeros(self.y.shape)       

    def feedforward(self):
        self.layer1 =sigmoid(numpy.dot(self.input,self.weights1))
        self.output =sigmoid(numpy.dot(self.layer1,self.weights2))
        
    def backprop(self):
        error_out=(self.y-self.output)*sigmoid_derivative(self.output)
        d_weights2 =numpy.dot(self.layer1.T,error_out)
        
        e_weights2 =numpy.dot(error_out,self.weights2.T)*sigmoid_derivative(self.layer1)  #distribute the error answer
        d_weights1 =numpy.dot(self.input.T,e_weights2)
        
        self.weights1 +=0.6*d_weights1
        self.weights2 +=0.6*d_weights2

if __name__ =="__main__":
    
    x =numpy.array([[0,0,1],
                   [0,1,1],
                   [1,0,1],
                   [1,1,1]])
              
    y =numpy.array([[0],[1],[1],[0]])
    
    nn=NeuralNetwork(x,y)

    for i in range(3000):
        nn.feedforward()
        nn.backprop()

    print(nn.output)
