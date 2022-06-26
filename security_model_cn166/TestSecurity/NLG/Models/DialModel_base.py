from abc import abstractmethod
class DialModelBase():
    def __init__(self):
        pass
    @abstractmethod
    def forward(self, input):
        print("ModelBase should not forward!")
        assert 1 == 0
        
