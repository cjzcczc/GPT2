import math
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = _children
        self._op = _op
        self.grad = 0.0
        self._backward = lambda: None  # Placeholder for backward function
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        out._backward = Value._add_backward(out,self,other)
        return out
    
    @classmethod
    def _add_backward(cls, out,a,b):
        def _backward():
            a.grad += out.grad
            b.grad += out.grad
        return _backward  # 返回函数本身
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        out._backward = Value._mul_backward(out,self,other)
        return out
    
    @classmethod
    def _mul_backward(cls, out,a,b):
        def _backward():
            a.grad += out.grad * b.data
            b.grad += out.grad * a.data
        return _backward
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Power must be int or float"
        out = Value(self.data ** other, (self,), f'**{other}')
        out._backward = Value._pow_backward(out,self,other)
        return out
    
    @classmethod
    def _pow_backward(cls, out,a,n):
        def _backward():
            a.grad += out.grad * n * (a.data ** (n - 1))
        return _backward
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        out._backward = Value._relu_backward(out,self)
        return out
    
    @classmethod
    def _relu_backward(cls, out,input):
        def _backward():
            input.grad += (out.data > 0) * out.grad
        return _backward
    
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)  # 使用 math.exp 计算 tanh
        out = Value(t, (self,), 'tanh')
        out._backward = Value._tanh_backward(out,self)
        return out
    
    @classmethod
    def _tanh_backward(cls, out,input):
        def _backward():
            input.grad += (1 - out.data ** 2) * out.grad
            input.grad
        return _backward
    
    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
