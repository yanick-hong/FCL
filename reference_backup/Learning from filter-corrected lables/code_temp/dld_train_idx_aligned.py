# dld_train_idx_aligned.py
import os, argparse, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
import torch.utils.data as data
import torchvision
from tqdm import tqdm

# ====== 来自 DLD 仓库的模块 ======
from utils.ema import EMA
from utils.vit_wrapper import vit_img_wrap
from utils.cifar_data_utils import Custom_dataset, Double_dataset
from utils.model_ResNet import ResNet_encoder
from utils.model_SimCLR import SimCLR_encoder
from utils.learning import adjust_learning_rate, cnt_agree
from utils.directional_diffusion_model import DirectionalConditionalModel, DirectionalDiffusion
from utils.add_ccn_noise import add_noise
from utils.log_config import setup_logger

from utils.pre_correction import precorrect_labels_in_two_view  # 你的文件里确实有
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

@torch.no_grad()
def prepare_2_fp_x_local(fp_encoder, train_dataset, device, fp_dim: int, num_workers: int = 4, batch_size: int = 200):
    """
    为整套训练样本预计算两路增广的嵌入：
      weak_embed[i]  = fp_encoder(x_weak_of_sample_i)
      strong_embed[i]= fp_encoder(x_strong_of_sample_i)
    """
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    N = len(train_dataset)
    weak_embed  = torch.zeros((N, fp_dim), device=device)
    strong_embed= torch.zeros((N, fp_dim), device=device)

    fp_encoder.eval()
    for xw, xs, _, indices in loader:
        xw = xw.to(device); xs = xs.to(device)
        ew = fp_encoder(xw)   # 你的 ViT 包装器会输出 (B, D)
        es = fp_encoder(xs)
        weak_embed[indices]   = ew
        strong_embed[indices] = es
    return weak_embed, strong_embed


def cast_label_to_one_hot_and_prototype(y: torch.Tensor, n_class: int) -> torch.Tensor:
    """
    若 y 已是 (N,C) 概率/one-hot 就直接返回；若 y 是 (N,) 则转成 one-hot。
    """
    if y.dim() > 1:
        return y
    return F.one_hot(y.to(torch.int64), num_classes=n_class).to(torch.float32)


def build_subset_cifar(dataset_name, device):
    if dataset_name == 'cifar10':
        n_class = 10
        train_raw = torchvision.datasets.CIFAR10(root='./data/', train=True, download=True)
        test_raw  = torchvision.datasets.CIFAR10(root='./data/', train=False, download=True)
        CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR_STD  = (0.2023, 0.1994, 0.2010)
    elif dataset_name == 'cifar100':
        n_class = 100
        train_raw = torchvision.datasets.CIFAR100(root='./data/', train=True, download=True)
        test_raw  = torchvision.datasets.CIFAR100(root='./data/', train=False, download=True)
        CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
        CIFAR_STD  = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError("dataset must be cifar10 or cifar100")
    return n_class, train_raw, test_raw, CIFAR_MEAN, CIFAR_STD

@torch.no_grad()
def evaluate(diffusion_model, loader, device, desc="eval"):
    diffusion_model.model.eval()
    diffusion_model.fp_encoder.eval()
    correct, total = 0, 0
    for _, batch in tqdm(enumerate(loader), total=len(loader), desc=desc, ncols=100):
        images, target, *_ = batch
        target = target.to(device)
        # 使用与原 test() 相同的生成-判定接口
        y_hat = diffusion_model.ddim_sample(x_batch=images, y_input=0, fp_x=None, last=True, stochastic=False).detach().cpu()
        correct += cnt_agree(y_hat, target.cpu())
        total   += images.shape[0]
    return 100.0 * correct / total

@torch.no_grad()
def export_pseudo_full_train(diffusion_model, raw_train_loader, device, save_path):
    diffusion_model.model.eval()
    diffusion_model.fp_encoder.eval()
    probs = []
    for _, batch in tqdm(enumerate(raw_train_loader), total=len(raw_train_loader), desc="export pseudo", ncols=100):
        images, *_ = batch
        # 这里返回的是每样本的 t=0 预测（通常为 one-hot 或概率向量）
        y_hat = diffusion_model.ddim_sample(x_batch=images, y_input=0, fp_x=None, last=True, stochastic=False).detach().cpu()
        probs.append(y_hat)
    probs = torch.cat(probs, dim=0).numpy()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, probs)

def main():
    parser = argparse.ArgumentParser()
    # 基础
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--noise_type', type=str, default='cifar100-idn-0.4')  # eg: cifar100-idn-0.4 / cifar100-sym-0.4
    parser.add_argument('--nepoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--log_name', type=str, default='cifar100-idn-0.4.aligned.log')

    # 划分对齐
    parser.add_argument('--train_idx', type=str, required=True, help='path to 45k train indices .npy')
    parser.add_argument('--val_idx',   type=str, required=True, help='path to 5k  val   indices .npy')

    # DLD/特征
    parser.add_argument('--num_models', type=int, default=2)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--loss_w', action='store_true', default=True)
    parser.add_argument('--to_single_label', action='store_true', default=False)
    parser.add_argument('--one_view', action='store_true', default=False)
    parser.add_argument('--use_cos', action='store_true', default=True)
    parser.add_argument('--ddim_n_step', type=int, default=10)
    parser.add_argument('--diff_encoder', type=str, default='resnet34')
    parser.add_argument('--objective', type=str, default='pred_res_noise')

    parser.add_argument('--fp_encoder', type=str, default='ViT', choices=['ViT','ResNet','SimCLR'])
    parser.add_argument('--ViT_type', type=str, default='ViT-L/14')
    parser.add_argument('--ResNet_type', type=str, default='resnet34')

    # 可选：周期性导出软标签
    parser.add_argument('--export_pseudo_every', type=int, default=0, help='>0 to export every N epochs on train(45k)')
    parser.add_argument('--export_dir', type=str, default='./exports/dld_pseudo')

    args = parser.parse_args()
    logger = setup_logger(args)

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = args.device
    dataset = args.noise_type.split('-')[0]   # cifar10 / cifar100
    n_class, train_raw, test_raw, CIFAR_MEAN, CIFAR_STD = build_subset_cifar(dataset, device)

    # ====== 索引对齐：从 50k 训练集中切出 45k/5k ======
    train_idx = np.load(args.train_idx)  # shape (45000,)
    val_idx   = np.load(args.val_idx)    # shape ( 5000,)

    train_data   = train_raw.data[train_idx]
    train_labels = np.array(train_raw.targets)[train_idx].tolist()

    val_data   = train_raw.data[val_idx]
    val_labels = np.array(train_raw.targets)[val_idx].tolist()

    # ====== fp encoder ======
    if args.fp_encoder == 'SimCLR':
        fp_dim = 2048
        state = torch.load(f'./model/SimCLR_128_{dataset}.pt', map_location=device)
        fp_encoder = SimCLR_encoder(feature_dim=128).to(device)
        fp_encoder.load_state_dict(state, strict=False)
    elif args.fp_encoder == 'ViT':
        fp_encoder = vit_img_wrap(args.ViT_type, device, center=CIFAR_MEAN, std=CIFAR_STD)  # 已内置归一化
        fp_dim = fp_encoder.dim
    else:  # ResNet
        fp_dim = 512 if args.ResNet_type == 'resnet34' else 2048
        fp_encoder = ResNet_encoder(feature_dim=fp_dim, base_model=args.ResNet_type).to(device)

    # ====== 数据集与加载器 ======
    from utils.ws_augmentation import TransformFixMatch_CIFAR10
    transform_fixmatch = TransformFixMatch_CIFAR10(CIFAR_MEAN, CIFAR_STD, 2, 10)

    train_dataset = Double_dataset(data=train_data, targets=train_labels, transform_fixmatch=transform_fixmatch)
    val_dataset   = Custom_dataset(val_data,  val_labels)
    test_dataset  = Custom_dataset(test_raw.data, test_raw.targets)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = data.DataLoader(val_dataset,   batch_size=200,            shuffle=False, num_workers=args.num_workers)
    test_loader  = data.DataLoader(test_dataset,  batch_size=200,            shuffle=False, num_workers=args.num_workers)

    # 可选：原始（未增强）训练集 loader 用于导出 pseudo
    raw_train_dataset = Custom_dataset(train_data, train_labels)
    raw_train_loader  = data.DataLoader(raw_train_dataset, batch_size=200, shuffle=False, num_workers=args.num_workers)

    # ====== 噪声标签注入（与原版一致）======
    noise_class = args.noise_type.split('-')[1]
    noise_ratio = float(args.noise_type.split('-')[2])
    if noise_class == 'idn':
        if args.noise_type.endswith('-0.0'):
            print('Training on pure label')
        else:
            noise_label = np.load(f'./noise_label_IDN/{args.noise_type}.npy')
            train_dataset.update_label(noise_label[:len(train_labels)])  # 对齐 45k
            print(f'Training on {args.noise_type} label noise')
        noisy_labels = torch.tensor(train_dataset.targets).squeeze().to(device)
    elif noise_class in ['sym','asym']:
        noisy_targets = add_noise(train_dataset.targets, noise_ratio, n_class, seed=None, symmetric_noise=(noise_class=='sym'))
        train_dataset.update_label(noisy_targets)
        noisy_labels = torch.tensor(train_dataset.targets).squeeze().to(device)
        print(f'Training on {args.noise_type} label noise')
    else:
        raise ValueError("Unknown noise type")

    # ====== 预计算 fp(x)（两路增广嵌入）======
    data_dir = './data/cifar-100-python' if dataset=='cifar100' else './data/cifar-10-batches-py'
    embed_dir = os.path.join(data_dir, 'fp_embed_train_cifar_aligned')
    print('Pre-computing fp embeddings for weak/strong views ...')
    weak_embed, strong_embed = prepare_2_fp_x_local(
    fp_encoder, train_dataset, device=device, fp_dim=fp_dim, num_workers=args.num_workers, batch_size=200
)

    weak_embed  = weak_embed.to(device)
    strong_embed= strong_embed.to(device)

    # ====== 构建 DLD 模型 ======
    model_path = f'./model/DLD_{args.fp_encoder}_{args.noise_type}.aligned.pt'
    base_model = DirectionalConditionalModel(
        n_steps=1000, y_dim=n_class, fp_dim=fp_dim, feature_dim=args.feature_dim,
        guidance=True, num_models=args.num_models, objective=args.objective, encoder_type=args.diff_encoder
    ).to(device)
    diffusion_model = DirectionalDiffusion(
        model=base_model, fp_encoder=fp_encoder, num_models=args.num_models, num_timesteps=1000,
        n_class=n_class, fp_dim=fp_dim, device=device, feature_dim=args.feature_dim, encoder_type=args.diff_encoder,
        objective=args.objective, sampling_timesteps=args.ddim_n_step, condition=True,
        convert_to_ddim=False, sum_scale=1., ddim_sampling_eta=0., beta_schedule='cosine'
    )
    diffusion_model.fp_encoder.eval()

    # ====== 优化器与 EMA（原版逻辑）======
    diffusion_loss = nn.MSELoss(reduction='none')
    max_val, best_test = 0.0, 0.0

    if diffusion_model.num_models == 1:
        optimizer = optim.Adam(diffusion_model.model.parameters(), lr=1e-4, weight_decay=0.0, betas=(0.9,0.999))
        ema_helper = EMA(mu=0.999); ema_helper.register(diffusion_model.model)
    else:
        optimizer_res   = optim.Adam(diffusion_model.model0.parameters(), lr=1e-4, weight_decay=0.0, betas=(0.9,0.999))
        optimizer_noise = optim.Adam(diffusion_model.model1.parameters(), lr=1e-4, weight_decay=0.0, betas=(0.9,0.999))
        ema_res, ema_noise = EMA(mu=0.999), EMA(mu=0.999)
        ema_res.register(diffusion_model.model0); ema_noise.register(diffusion_model.model1)

    print(f'[Aligned] Training DLD on {dataset} with {args.noise_type}. ckpt -> {model_path}')
    logger = setup_logger(args)

    for epoch in range(args.nepoch):
        if diffusion_model.num_models == 1:
            diffusion_model.model.train()
        else:
            diffusion_model.model0.train(); diffusion_model.model1.train()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train epoch {epoch}', ncols=120)
        for i, batch in pbar:
            xw, xs, y_noisy, data_indices = batch[:4]
            xw, xs = xw.to(device), xs.to(device)
            y_noisy = y_noisy.to(device)

            # 使用预计算嵌入
            fp_w = weak_embed[data_indices, :].to(device)
            fp_s = strong_embed[data_indices, :].to(device)

            # two-view 预校正
            y_w, y_s, lw_w, lw_s, y_n, gamma = precorrect_labels_in_two_view(
                fp_embd_w=fp_w, fp_embd_s=fp_s, y_noisy=y_noisy,
                weak_embed=weak_embed, strong_embed=strong_embed,
                noisy_labels=noisy_labels, k=args.k, n_class=n_class,
                use_cosine_similarity=args.use_cos, to_single_label=False  # 关键：要分布
            )


            x_batch = xw if args.one_view else (1 - gamma.view(-1,1,1,1))*xw + gamma.view(-1,1,1,1)*xs
            # one-hot 化
            if y_w.dim()==1:
                y_onehot_w = cast_label_to_one_hot_and_prototype(y_w.to(torch.int64), n_class=n_class)
            else:
                y_onehot_w = y_w
            y0 = cast_label_to_one_hot_and_prototype(y_w, n_class=n_class).to(device)

            zeros = torch.zeros_like(y0)

            # 余弦式 warmup lr
            progress = i/len(train_loader)+epoch
            if diffusion_model.num_models == 1:
                adjust_learning_rate(optimizer, progress, args.warmup_epochs, args.nepoch, lr_input=1e-3)
            else:
                adjust_learning_rate(optimizer_res, progress, args.warmup_epochs, args.nepoch, lr_input=1e-3)
                adjust_learning_rate(optimizer_noise, progress, args.warmup_epochs, args.nepoch, lr_input=1e-3)

            # 采样 t
            n = x_batch.size(0)
            t = torch.randint(low=0, high=diffusion_model.num_timesteps, size=(n//2+1,), device=device)
            t = torch.cat([t, diffusion_model.num_timesteps-1-t], dim=0)[:n]

            # 前向 + 损失
            output, e = diffusion_model.forward_t(zeros, y0, x_batch, t, fp_w)
            if diffusion_model.objective == 'pred_res_noise':
                L_res  = diffusion_loss(output[0], e[0])
                L_noise= diffusion_loss(output[1], e[1])
                wL_res   = torch.matmul(lw_w, L_res)  if args.loss_w else L_res
                wL_noise = torch.matmul(lw_w, L_noise) if args.loss_w else L_noise
                l_res  = torch.mean(wL_res)
                l_noise= torch.mean(wL_noise)
                if diffusion_model.num_models == 2:
                    optimizer_res.zero_grad();   l_res.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(diffusion_model.model0.parameters(), 1.0); optimizer_res.step();   ema_res.update(diffusion_model.model0)
                    optimizer_noise.zero_grad(); l_noise.backward()
                    torch.nn.utils.clip_grad_norm_(diffusion_model.model1.parameters(), 1.0); optimizer_noise.step(); ema_noise.update(diffusion_model.model1)
                    pbar.set_postfix({'res_loss': l_res.item(), 'noise_loss': l_noise.item()})
                else:
                    loss = 0.1*l_res + 0.9*l_noise
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0); optimizer.step(); ema_helper.update(diffusion_model.model)
                    pbar.set_postfix({'loss': loss.item()})
            else:
                L_noise = diffusion_loss(output, e)
                wL_noise = torch.matmul(lw_w, L_noise) if args.loss_w else L_noise
                loss = torch.mean(wL_noise)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0); optimizer.step(); ema_helper.update(diffusion_model.model)
                pbar.set_postfix({'loss': loss.item()})

        # ====== 验证选择最优，再测试 ======
        if epoch >= args.warmup_epochs:
            val_acc  = evaluate(diffusion_model, val_loader, device, desc='val')
            logger.info(f"epoch {epoch} | val_acc={val_acc:.2f}%")
            if val_acc > max_val:
                max_val = val_acc
                test_acc = evaluate(diffusion_model, test_loader, device, desc='test')
                best_test = test_acc
                # 存 ckpt
                states = {}
                if diffusion_model.num_models == 1:
                    states['model'] = diffusion_model.model.state_dict()
                else:
                    states['model0'] = diffusion_model.model0.state_dict()
                    states['model1'] = diffusion_model.model1.state_dict()
                states['fp_encoder'] = diffusion_model.fp_encoder.state_dict()
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(states, model_path)
                logger.info(f"[BEST] epoch {epoch} | val={val_acc:.2f}% | test={test_acc:.2f}% -> save {model_path}")

        # ====== 可选：导出训练集的软标签（供方案3）======
        if args.export_pseudo_every > 0 and ((epoch+1) % args.export_pseudo_every == 0):
            save_p = os.path.join(args.export_dir, f"pseudo_train45k_epoch{epoch+1:03d}.npy")
            export_pseudo_full_train(diffusion_model, raw_train_loader, device, save_p)
            logger.info(f"Exported train(45k) pseudo to {save_p}")

    print(f"Training done. Best val={max_val:.2f}%, corresponding test={best_test:.2f}%")

if __name__ == "__main__":
    main()
