import torch
from torch.utils.data import DataLoader
from mydataset import MyDataset, custom_collate_fn
from model import TSLSTM
from snntorch import functional as SF
from tqdm import tqdm
import pathlib
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 233
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

trainloader = DataLoader(
    MyDataset("cached.json", use_cached=True),  # warning, if not using cached, it will take an extremely long time
    batch_size=64,
    shuffle=True,
    collate_fn=custom_collate_fn
)


#  Initialize Network
net = TSLSTM(device, 24576, 2560)


def run(net, trainloader, num_epochs, from_pretrained=False):  # epoch here means total epoch including trained
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(num_classes=24576, correct_rate=0.7, incorrect_rate=0.3)
    # load previous model
    if from_pretrained is True and pathlib.Path("model\\translation\\log.csv").exists():
        net.load_state_dict(torch.load("model\\translation\\net.pth"))
        optimizer.load_state_dict(torch.load("model\\translation\\optimizer.pth "))
        # reclaim previous checkpoint parameters
        train_log = pd.read_csv("model\\translation\\log.csv")
        trained_epochs = int(train_log.iloc[-1]["epoch"])
        trained_batches = int(train_log.iloc[-1]["batch"])
        loss_mean = train_log.iloc[-1]["loss_mean"]
        acc_mean = train_log.iloc[-1]["acc_mean"]
    else:
        # else init
        trained_epochs, trained_batches, loss_mean, acc_mean = 1, 0, 0.0, 0.0
        train_log = pd.DataFrame(columns=["epoch", "batch", "loss", "loss_mean", "acc", "acc_mean"])
        train_log.to_csv("model\\translation\\log.csv", index=False)
    # run epoch
    for current_epoch in range(trained_epochs, num_epochs+1):
        print(f'Epoch [{current_epoch}/{num_epochs}]')
        fp = open('model\\translation\\log.csv', 'a')
        # training loop
        loop = tqdm(
            trainloader,
            unit='batches',
            bar_format='{l_bar}{bar:32}{r_bar}'
        )
        for current_batch, (data, target) in enumerate(loop, start=trained_batches):
            data = data.to(device)
            target = target.to(device)
            net.train()
            spikes = net.forward(data)
            loss = loss_fn(spikes, target)
            # gradient calculation + weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # calculate acc and loss
            acc = SF.accuracy_rate(spikes, target)
            acc_mean = (acc_mean*current_batch + acc) / (current_batch+1)
            loss = loss.item()
            loss_mean = (loss_mean*current_batch + loss) / (current_batch+1)
            torch.cuda.empty_cache()
            # show info
            loop.set_description(f'Train [{current_epoch}/{num_epochs}]')
            loop.set_postfix(
                loss_mean=f'{loss_mean:.6f}',
                acc_mean=f'{(acc_mean*100):.6f}%'
            )
            # write log and save model
            fp.write(f'{current_epoch}, {current_batch}, {loss}, {loss_mean}, {acc}, {acc_mean}\n')
            fp.flush()
            if current_batch % 100 == 0 and current_batch != 0:
                # save model and optimizer
                torch.save(net.state_dict(), "model\\translation\\net.pth")
                torch.save(optimizer.state_dict(), "model\\translation\\optimizer.pth")
        # reset checkpoint parameters
        trained_batches, loss_mean, acc_mean = 0, 0.0, 0.0
        # close file point
        fp.close()
        # save model and optimizer when completed
        torch.save(net.state_dict(), "model\\translation\\net.pth")
        torch.save(optimizer.state_dict(), "model\\translation\\optimizer.pth")


run(net, trainloader, num_epochs=1, from_pretrained=True)
