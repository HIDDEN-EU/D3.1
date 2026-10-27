import torch
def get_shape(x):
    if type(x)==torch.Tensor:
        return "type {}, shhape {}".format(type(x) , x.shape)
    elif x==None:
        return "got none"
    elif type(x)==list:
        return "got list , len {} and 1st elem type {}".format(len(x) , type(x[0]))
    
    