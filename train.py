import torch
import os
import torch.nn as nn
from dataset import *

from dataset import *
from loss import *


def trainer(model, category, config):
    '''
    Training the UNet model
    :param model: the UNet model
    :param category: the category of the dataset
    '''
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=config.model.learning_rate)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay
    )
    train_dataset = Dataset_maker(
        root= config.data.data_dir,
        category=category,
        config = config,
        is_train=True,
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True,
    )
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(config.model.checkpoint_dir):
        os.mkdir(config.model.checkpoint_dir)


    for epoch in range(config.model.epochs):
        losses = []
        for step, batch in enumerate(trainloader):
            if step == 200:
                break
            optimizer.zero_grad()
            # loss = 0
            # for _ in range(2):
            t = torch.randint(0, config.model.trajectory_steps, (batch[0].shape[0],), device=config.model.device).long()
            # print(f'Batch :: {batch[0].shape}')
            # print(f"t shape :: {t.shape}")
            # print(f"t :: {t}")
            loss = get_loss(model, batch[0], t, config) 
            loss.backward()
            optimizer.step()
            # if (epoch+1) % 25 == 0 and step == 0:
            print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item()}")
            losses.append(loss.item())
            if (epoch+1) %250 == 0 and epoch>0 and step ==0:
                if config.model.save_model:
                    model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, category)
                    if not os.path.exists(model_save_dir):
                        os.mkdir(model_save_dir)
                    torch.save(model.state_dict(), os.path.join(model_save_dir, str(epoch+1)))
        print(f"Epoch {epoch+1}/{config.model.epochs} | Average Loss: {sum(losses)/len(losses)}")
                
