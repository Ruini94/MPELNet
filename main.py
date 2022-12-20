import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import time
import argparse
from model import Dehaze, Discriminator
from MyDataset import MyDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_msssim import msssim, ssim
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from ptflops import get_model_complexity_info
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.fastest = True

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=r"G:\big_dataset\deraze_dataset\D_dehaze\RESIDE-6K")
parser.add_argument('--data_name', type=str, default='RESIDE-6K')
parser.add_argument('--cropSize', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--model_path', type=str, default="best_models")
parser.add_argument('--nEpochs', type=int, default=200)
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
args = parser.parse_args()

os.makedirs("images", exist_ok=True)
os.makedirs("best_models", exist_ok=True)

learning_rate = args.learning_rate

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MyEnsembleNet = Dehaze()
print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))
DNet = Discriminator()
print('# Discriminator parameters:', sum(param.numel() for param in DNet.parameters()))

G_optimizer = torch.optim.Adam(MyEnsembleNet.parameters(), lr=0.0001)
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[args.nEpochs * 0.5, args.nEpochs * 0.7,
                                                                            args.nEpochs * 0.8], gamma=0.5)
D_optim = torch.optim.Adam(DNet.parameters(), lr=0.0001)
scheduler_D = torch.optim.lr_scheduler.MultiStepLR(D_optim, milestones=[args.nEpochs * 0.5, args.nEpochs * 0.7,
                                                                        args.nEpochs * 0.8], gamma=0.5)

dataset_train = MyDataset(root=args.data_path, name=args.data_name, cropSize=args.cropSize, mode="train")
train_loader = DataLoader(dataset=dataset_train, num_workers=1, batch_size=args.batch_size, shuffle=True,
                          drop_last=True)
test1_set = MyDataset(root=args.data_path, name=args.data_name, cropSize=256, mode="test")
test_loader = DataLoader(test1_set, batch_size=4, shuffle=False)

MyEnsembleNet = MyEnsembleNet.to(device)
MyEnsembleNet = torch.nn.DataParallel(MyEnsembleNet, device_ids=device_ids)
DNet = DNet.to(device)
DNet = torch.nn.DataParallel(DNet, device_ids=device_ids)

msssim_loss = msssim


def save_checkpoint(model, epoch, psnr):
    model_out_path = os.path.join(args.model_path,
                                  "{}_epoch_{}_PSNR_{:.4f}.pth".format(args.data_name, epoch, psnr))

    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(args.model_path))


def validate(epoch):
    MyEnsembleNet.eval()
    with torch.no_grad():
        haze_img, _ = next(iter(test_loader))
        haze_img = haze_img.cuda()
        res = MyEnsembleNet(haze_img)
        img_sample = torch.cat((haze_img, res), 0)
        save_image(img_sample, "images/%s.png" % epoch, nrow=4, normalize=True)


if __name__ == "__main__":
    best_psnr = 0.0
    iteration = 0
    for epoch in range(args.nEpochs):
        start_time = time.time()
        scheduler_G.step()
        scheduler_D.step()

        MyEnsembleNet.train()
        DNet.train()

        print(epoch)
        for batch_idx, (hazy, clean) in enumerate(train_loader):
            iteration += 1
            hazy = hazy.to(device)
            clean = clean.to(device)

            output = MyEnsembleNet(hazy)

            DNet.zero_grad()
            real_out = DNet(clean).mean()  # 这个mean是 minibatch个数取均值
            fake_out = DNet(output).mean()
            D_loss = 1 - real_out + fake_out

            D_loss.backward(retain_graph=True)
            MyEnsembleNet.zero_grad()
            adversarial_loss = torch.mean(1 - fake_out)
            smooth_loss_l1 = F.smooth_l1_loss(output, clean)
            msssim_loss_ = -msssim_loss(output, clean, normalize=True)

            total_loss = smooth_loss_l1 + 0.0005 * adversarial_loss + 0.5 * msssim_loss_
            
            total_loss.backward()
            D_optim.step()
            G_optimizer.step()

        psnr = 0
        MyEnsembleNet.eval()
        for i in range(test1_set.__len__()):
            val_data, val_label = test1_set[i]
            val_data = val_data.unsqueeze(0).cuda()
            val_label_numpy = val_label.numpy()
            with torch.no_grad():
                val_out = MyEnsembleNet(val_data)
            val_out = val_out.cpu().data[0].numpy()
            psnr += compare_psnr(val_out, val_label_numpy, data_range=1.0)

        psnr = psnr / (i + 1)

        if psnr > best_psnr:
            best_psnr = psnr
            save_checkpoint(MyEnsembleNet, epoch, best_psnr)

        validate(epoch)
