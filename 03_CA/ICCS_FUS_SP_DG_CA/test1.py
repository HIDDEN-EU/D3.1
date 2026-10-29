import torch
radius = 2
yy, xx = torch.meshgrid(
    torch.arange(-radius, radius + 1, ),
    torch.arange(-radius, radius + 1, ),
)

print(yy)
print(xx)

gaussian = torch.exp(-(xx**2 + yy**2) / (2 * 1.0**2))

print(gaussian.shape)
print(gaussian[0:5,0:5])