from engine import *
import numpy as np


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []

class neuron(Module):
    def __init__(self, nin, activation = "relu"): # nin is the number of inputs of this neuron
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(nin)] # will initialize the weights to a random unifrom number between -1, 1 with dimension of the nin (as we have a weight for every input)
        self.b = Value(np.random.uniform(-1,1))
        self.activation = activation
    
    # this method will be called when we call the object as a function (we will call it with the inputs x during backprop)
    def __call__(self,x):
        z = sum((xi * wi for xi, wi in zip(x, self.w)), self.b) # start is the initial value of the sum, so we start with the bias
        if self.activation == "relu":
            a = z.relu()
        elif self.activation == "tanh":
            a = z.tanh()
        return a
    
    # let us gather the parameters of the NN so that later we can operate on them simultaneuosly (optimise them), and nudge every one of them to minimize the loss
    # Tensor has the same thing but it returns parameter tensors for each nn module, for us its the parameter scalars
    def parameters(self):
        return self.w + [self.b] # returns the concatenated w and b lists 

    def __repr__(self):
        return f"{self.activation} Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, activation = "relu"): # nin is the dimension of the input to each  neuron and nout is the number of neurons (they make up the number of outputs)
        self.neurons = [neuron(nin, activation=activation) for _ in range(nout)]

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        params = [p for neuron in self.neurons for p in neuron.parameters()]
        # # will concatenate the parameters of the single neurons
        # for neuron in self.neurons:
        #     params.extend(neuron.parameters) # extend instead of append because append will append the whole list as one item, but extend extends params with the list items individually
        return params
    
    def __repr__(self):
        return f"Layer of [{", ".join(str(n) for n in self.neurons)}]"
    
class mlp(Module):
    def __init__(self, nin, nouts, activations): # instead of the number of neurons for a single layer, now we do it for several layers
        sizes = [nin] + nouts # concatenate nin to the beginning of the nouts (as nin is the number of neurons in the input layer)
        self.layers = [Layer(sizes[i],sizes[i+1],activation=activations[i]) for i in range(len(sizes) - 1)]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x
    
    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{" ,".join(str(layer) for layer in self.layers)}]"