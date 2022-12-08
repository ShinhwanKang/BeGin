import torch



class Binarizer(torch.autograd.Function):
    """
        Binarizes {0, 1} a real valued tensor.
    """
    def forward(self, inputs, threshold):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput

class Ternarizer(torch.autograd.Function):
    """
        Ternarizes {-1, 0, 1} a real valued tensor.
    """
    def forward(self, inputs, threshold):
        outputs = inputs.clone()
        outputs.fill_(0)
        outputs[inputs < 0] = -1
        outputs[inputs > threshold] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput