from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

aug_list = []
# aug_list.append(transforms.RandomCrop(size=[220,500]))
# aug_list.append(transforms.RandomAdjustSharpness(sharpness_factor=2,p=0.3))
aug_list.append(transforms.ColorJitter(brightness=0., contrast=0., hue=0., saturation=0.8))
aug_list.append(transforms.RandomAffine(degrees=[-20., 20.], scale=[0.8, 1.2]))
aug_list.append(transforms.RandomHorizontalFlip(0.5))
aug_list.append(transforms.RandomVerticalFlip(0.5))
aug_list.append(transforms.Resize((512, 512)))
# aug_list.append(transforms.ToTensor())

transform = transforms.Compose(aug_list)

# test image
while(True):
    image = transform(Image.open(r'C:\Users\s.nakamura\workspace\projects\shimaseiki\datasets\split_anomaly\tmp34_part16.png'))
    plt.imshow(image)
    plt.show()
    i = input()
    if i == 'q':
        break
