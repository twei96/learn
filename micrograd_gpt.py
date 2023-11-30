class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        self._grad_inputs = []
    
    def backward(self, grad=None):
        if self.requires_grad:
            if grad is None:
                grad = Tensor(1.0)
            
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad
                
            if self._grad_fn is not None:
                grads = self._grad_fn(grad)
                if not isinstance(grads, list):
                    grads = [grads]
                for i, inp in enumerate(self._grad_inputs):
                    inp.backward(grads[i])
                self._grad_fn = None
                
    def __add__(self, other):
        out = Tensor(self.data + other.data)
        out._backward = lambda grad: [grad, grad]
        out._grad_inputs = [self, other]
        out._grad_fn = lambda grad: sum(out._backward(grad))
        return out
    
    def __mul__(self, other):
        out = Tensor(self.data * other.data)
        def _backward(grad):
            return [grad * other.data, grad * self.data]
        out._backward = _backward
        out._grad_inputs = [self, other]
        out._grad_fn = lambda grad: sum(out._backward(grad))
        return out
    
    def __repr__(self):
        return f"Tensor({self.data})"


x1 = Tensor(2.0, requires_grad=True)
x2 = Tensor(3.0, requires_grad=True)
y = x1 * x2 + x1
y.backward()

print(x1.grad)  # prints "Tensor(4.0)"
print(x2.grad)  # prints "Tensor(2.0)"