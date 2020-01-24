import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import json
import matplotlib.pyplot as plt

torch.manual_seed(25)

model = models.resnet101(pretrained=True)
model.eval()

CLASSES = json.load(open('./imagenet_samples/imagenet_classes.json'))
idx2class = [CLASSES[str(i)] for i in range(1000)]

img = Image.open('imagenet_samples/corgie.jpg')

img_transforms = transforms.Compose([
    transforms.Resize((224,224), Image.BICUBIC),
    transforms.ToTensor(),
    ])

img_tensor = img_transforms(img)
img_tensor = img_tensor.unsqueeze(0)

print('Image Tensor Shape is ', img_tensor.size())

original_img_view = img_tensor.squeeze(0).detach()
original_img_view = original_img_view.transpose(0,2).transpose(0,1).numpy()
plt.imshow(original_img_view)

# Before Attack, Check the Model Performance
output = model(img_tensor)
prediction = output.max(1, keepdim=False)[1]

prediction_idx = prediction.item()
prediction_name = idx2class[prediction_idx]

print('Label: {} | Name: {}'.format(prediction_idx, prediction_name))

# FGSM
def fgsm_attack(image, epsilon, gradient):
    sign_gradient = gradient.sign()
    perturbed_image = image + epsilon * sign_gradient
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Train
img_tensor.requires_grad_(True)
pred = model(img_tensor)
loss = F.nll_loss(pred, torch.tensor([263]))
model.zero_grad()
loss.backward()

# Attack
gradient = img_tensor.grad.data 
epsilon = 0.03
perturbed_data = fgsm_attack(img_tensor, epsilon, gradient)

output = model(perturbed_data)

# FGSM Perfomance Check
perturbed_prediction = output.max(1, keepdim=True)[1]
perturbed_prediction_idx = perturbed_prediction.item()
perturbed_prediction_name = idx2class[perturbed_prediction_idx]

print('After Attack, Label: {} | Name: {}'.format(perturbed_prediction_idx, perturbed_prediction_name))

# Image Check
f, a = plt.subplots(1,2, figsize = (10,16))
a[0].set_title(prediction_name)
a[0].imshow(original_img_view)

perturbed_data_view = perturbed_data.squeeze(0).detach()
perturbed_data_view = perturbed_data_view.transpose(0,2).transpose(0,1).numpy()

a[1].set_title(perturbed_prediction_name)
a[1].imshow(perturbed_data_view)

plt.show()
