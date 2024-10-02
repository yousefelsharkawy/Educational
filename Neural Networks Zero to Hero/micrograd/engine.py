import math

class Value:
    # initialization method
    def __init__(self ,data ,_children=() ,_op = "", label=""): # children is used to keep track of the nodes that produced our node, it is an empty tuple by default, and we will construct it in the operation methods
        self.data = data
        self.grad = 0 # we will use this variable to store the gradient of the node, it is initialized to zero which means the node doesn't affect the final outcome by changing it
        self._prev = set(_children) # that is done for efficiency
        self._op = _op # we will use this variable to store the operation used to create our node (it will be empty string by default) and we will construct it in the operation methods
        self.label = label # just to store the name of the variable for visualization purposes
        self._backward = lambda: None # it will be a function that will be used to take the gradient of the node -upstream- and propagate it to its previous, it is initialized to an empty function that doesnt do anything and it will stay like that for the leave nodes with no previous
    
    
    # representation method, which is called when the object is printed
    def __repr__(self):
        # if we did not have repr, it will print some ugly cryptic stuff about the location of the object in memory
        return f"Value:(data={self.data})"
    
    def __add__(self, other):
        # we want to support Value object additions (a + b) and also scalar additions (a + 1)
        other = other if isinstance(other,Value) else Value(other) # if other is a value object then keep it the same, and if not, wrap the other -which we will assume to be the data- in a Value object
        # create a value object of the sum and return it, when we write a + b python will internally call a.__add__(b)
        out = Value(self.data + other.data, (self, other) ,"+")
        # let's define the backward function when we cann the out object _backward. notice that we create a node through an operation and we create at the same time how to backward its gradient to the nodes that createed it (if we kept doing that then all the nodes will have backward implementation except the leaf nodes that are considered the beginning of the mathematical expression)
        def _backward():
            # we want to take out grad and propagate it to the previous nodes, and in addition we simply route the out grad to the previous nodes
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1

    def __sub__(self,other):
        return self + (-other) # mine: we did not put the backward code since we just call addition and it will handle it in there (so istead of codifying the special cases like subtraction, we make use of the general codified code of addition -addition to a negation of the number-), we did the same with division below (it is a special case of multiplication)
    
    def __rsub__(self,other):
        print("rsub is called")
        return (-self) + other # we did that because subraction does't have substitution property 

    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        # create a value object of the sum and return it, when we write a + b python will internally call a.__add__(b)
        out = Value(self.data * other.data, (self, other) ,"*")
        def _backward():
            # we do += because we want to accumulate the gradients in case a node is used in multiple operations (otherwise we will overwrite the gradient every single time), because when a variable is used in multiple paths, the gradient will be the sum of the gradients coming from these paths during backpropagation
            self.grad += out.grad * other.data 
            other.grad += out.grad * self.data
        out._backward = _backward
        return out
    
    def __rmul__(self, other): #other x self
        # print("self is ", self) a
        # print("other is ", other) 3 or whatever the number
        return self * other # we swapped the operands to be a * 2, so we basically called the __mul__ method of a
    
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), "exp")

        def _backward():
            self.grad += out.grad * out.data # out.data is e^x which is the derivative of e^x
        out._backward = _backward 
        return out
    
    def __pow__(self,other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = Value(self.data**other, (self, ), f"**{other}")

        def _backward():
            self.grad = out.grad * (other * (self.data**(other - 1)) )
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        # self / other -> self * (1/other) -> self * (other**-1), basically we just call the pow function in here
        return self * (other**-1)

    def __rtruediv__(self,other):
        return other * self**-1 # we did that because division doesnt have substitution propoerty (the otiginal forumla is other/self) 

    # we can create more complex functions (tanh instead of its basic components, such as exp mul div and so on) that the basic atomic operations (this is an example of scaling things up, we can create functions at arbitrary points of abstractions) and it is ok as long as we can code its derivative
    def tanh(self):
        x = self.data
        tanhx = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        out = Value(tanhx, (self, ), "tanh")
        def _backward():
            self.grad += out.grad * (1 - tanhx**2)
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self, ), "relu")
        def _backward():
            self.grad += out.grad * (out.data > 0)
        out._backward = _backward
        return out
    
    def backward(self):
        # this function will build the topological graph then call _backward() functions for the nodes in reverse
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                # a node is only added to the list of topo after we have gone through all of its children (you're only going to be in the list once all of your children are in the list, and that is how we guarantee the sort of the nodes from left to right)
                topo.append(v)

        build_topo(self) # we will start the topological sort from o
        self.grad = 1
        for node in reversed(topo):
            node._backward()

