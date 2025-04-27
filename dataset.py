import os
from glob import glob
from pathlib import Path
import shutil
import numpy as np
import csv
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10



class Dataset_maker(torch.utils.data.Dataset):
    def __init__(self, root, category, config, is_train=True):
        self.image_transform = transforms.Compose(
            [   
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((config.data.image_size, config.data.image_size)),  
                transforms.ToTensor(), # Scales data into [0,1] 
                # transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(), # Scales data into [0,1] 
            ]
        )
        self.config = config
        self.is_train = is_train
        if is_train:
            self.image_files = glob(
                os.path.join(root,"Clean data", "Train", "*.jpg")
            )
        else:
            self.image_files = []
            test_images = glob(os.path.join(root, "Clean data", "Test", "*.jpg"))
            # anom_images = glob(os.path.join(root, "Anomolous data", "*.jpg"))

            self.image_files.extend(test_images)
            # self.image_files.extend(anom_images)
        

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        # image = self.image_transform(image)
        image = self.image_transform(image)
        
        # if(image.shape[0] == 1):
        #     image = image.expand(3, self.config.data.image_size, self.config.data.image_size)
        if self.is_train:
            label = 'good'
            return image, label
        else:
            image_dir = os.path.dirname(image_file)
            if "Anomolous data" in image_dir:
                label = "defective"
                if self.config.data.mask:
                    # Replace .jpg with .png to find corresponding mask
                    mask_file = image_file.replace(".jpg", ".png")
                    if os.path.exists(mask_file):
                        mask = Image.open(mask_file)
                        target = self.mask_transform(mask)
            else:
                label = "good"
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            # 
            # if self.config.data.mask:
            #     if self.config.data.name == 'Bosch':
            #             mask_file = image_file.replace(
            #                 ".jpg", ".png"
            #             )
                        
            #             target = Image.open(
            #                 mask_file
            #             )
            #             target = self.mask_transform(target)
            #     label = 'defective'
            # else:
            #     if os.path.dirname(image_file).endswith("good"):
            #         target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            #         label = 'good'
            #     else :
            #         target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            #         label = 'defective'
            
            return image, target, label, image_file

    def __len__(self):
        return len(self.image_files)


# import os
# from glob import glob
# from pathlib import Path
# import shutil
# import numpy as np
# import csv
# import torch
# import torch.utils.data
# from PIL import Image
# from torchvision import transforms
# import torch.nn.functional as F
# import torchvision.datasets as datasets
# from torchvision.datasets import CIFAR10



# class Dataset_maker(torch.utils.data.Dataset):
#     def __init__(self, root, category, config, is_train=True):
#         self.image_transform = transforms.Compose(
#             [   transforms.Grayscale(num_output_channels=1),
#                 transforms.Resize((config.data.image_size, config.data.image_size)),  
#                 transforms.ToTensor(), # Scales data into [0,1] 
#                 transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
#             ]
#         )
#         self.config = config
#         self.mask_transform = transforms.Compose(
#             [   transforms.Grayscale(num_output_channels=1),
#                 transforms.Resize((config.data.image_size, config.data.image_size)),
#                 transforms.ToTensor(), # Scales data into [0,1] 
#             ]
#         )
#         if is_train:
#             if category:
#                 self.image_files = glob(
#                     os.path.join(root, category, "train", "good", "*.png")
#                 )
#             else:
#                 self.image_files = glob(
#                     os.path.join(root, "train", "good", "*.png")
#                 )
#         else:
#             if category:
#                 self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
#             else:
#                 self.image_files = glob(os.path.join(root, "test", "*", "*.png"))
#         self.is_train = is_train

#     def __getitem__(self, index):
#         image_file = self.image_files[index]
#         image = Image.open(image_file)
#         image = self.image_transform(image)
#         # if(image.shape[0] == 1):
#         #     image = image.expand(3, self.config.data.image_size, self.config.data.image_size)
#         if self.is_train:
#             label = 'good'
#             return image, label
#         else:
#             if self.config.data.mask:
#                 if os.path.dirname(image_file).endswith("good"):
#                     target = torch.zeros([1, image.shape[-2], image.shape[-1]])
#                     label = 'good'
#                 else :
#                     if self.config.data.name == 'MVTec':
#                         target = Image.open(
#                             image_file.replace("/test/", "/ground_truth/").replace(
#                                 ".png", "_mask.png"
#                             )
#                         )
#                     else:
#                         target = Image.open(
#                             image_file.replace("/test/", "/ground_truth/"))
#                     target = self.mask_transform(target)
#                     label = 'defective'
#             else:
#                 if os.path.dirname(image_file).endswith("good"):
#                     target = torch.zeros([1, image.shape[-2], image.shape[-1]])
#                     label = 'good'
#                 else :
#                     target = torch.zeros([1, image.shape[-2], image.shape[-1]])
#                     label = 'defective'
                
#             return image, target, label

#     def __len__(self):
#         return len(self.image_files)


# AUROC: (96.5,89.3)