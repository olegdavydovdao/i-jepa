import os
import sys; sys.path.append(".")
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from src.i_jepa.models import EncoderViT
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import destroy_process_group
from src.i_jepa.train import main

def main_probe():
    # LOAD SAVED TARGET_ENCODER
    (use_ddp, rank, local_rank, world_size, device, master_process,
     device_type, make_transform, mask_collator, dist_sampler,
     data_loader, cfg, batch_size_per_process, grad_accum_steps) = main(share=True)
    
    path_to_saved_dir = "checkpoints"
    path_to_saved_file = "checkpoint_end_pretrain.pt"
    path_to_pt = os.path.join(path_to_saved_dir, path_to_saved_file)
    saved_dict = torch.load(path_to_pt)
    saved_target_encoder_sd = saved_dict['target_encoder']

    target_encoder_vit = EncoderViT(cfg)
    target_encoder_vit.to(device)
    for p in target_encoder_vit.parameters():
        p.requires_grad = False

    major, minor = torch.cuda.get_device_capability()
    use_compile = cfg.use_compile and major >= 7
    # compile forward function of 3 separate models.
    if master_process:
        print(f"{use_compile=}")
    if use_compile:
        target_encoder_vit = torch.compile(target_encoder_vit)

    target_encoder_vit.load_state_dict(saved_target_encoder_sd)

    # LINEAR PROBE CLASSIFIER
    probe = nn.Linear(cfg.emb_dims, cfg.num_classes)
    nn.init.normal_(probe.weight, std = cfg.init_std)
    nn.init.zeros_(probe.bias)
    probe.to(device)
    if use_compile:
        probe = torch.compile(probe)
    if use_ddp:
        probe = DDP(probe, device_ids=[local_rank])

    probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=cfg.lr_eval, weight_decay=cfg.wd_eval, fused=cfg.use_adamw_fused)
    mask_collator.finetune = True
    target_encoder_vit.eval()

    scaler = torch.amp.GradScaler(device_type) if cfg.use_fp16 else None
    amp_dtype = torch.bfloat16 if cfg.use_bfloat16 else torch.float16
    enabled_amp = cfg.use_bfloat16 or cfg.use_fp16
    if master_process:
        print(f"{cfg.use_fp16=}")
        print(f"{cfg.use_bfloat16=}")

    def run_probe(dist_sampler_any, data_loader_any, use_probe_train):
        for epoch_eval in range(cfg.eval_num_epoch):
            dist_sampler_any.set_epoch(epoch_eval)
            correct = torch.tensor(0, device=device)
            total = torch.tensor(0, device=device)

            for step, (xb_proc, yb_proc) in enumerate(data_loader_any):
                if use_probe_train:
                    probe_optimizer.zero_grad()
                loss_eval_accum = 0.0
                for k in range(grad_accum_steps):
                    xb = xb_proc[k*cfg.batch_size: k*cfg.batch_size + cfg.batch_size]
                    yb = yb_proc[k*cfg.batch_size: k*cfg.batch_size + cfg.batch_size]
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)

                    if use_probe_train:
                        if use_ddp:
                            probe.require_backward_grad_sync = (k==grad_accum_steps-1)

                    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=enabled_amp):
                        with torch.no_grad():
                            s_y = target_encoder_vit(xb).mean(dim=1) # average pool
                        logits = probe(s_y)
                        loss = F.cross_entropy(logits, yb)
                        
                    loss = loss / grad_accum_steps
                    loss_eval_accum += loss.detach()

                    with torch.no_grad():
                        compare = logits.argmax(dim=-1) == yb
                        total += len(compare)
                        correct += compare.sum()

                    if use_probe_train:
                        if cfg.use_fp16:
                            scaler.scale(loss).backward()
                        else: loss.backward()
                if use_ddp:
                    dist.all_reduce(loss_eval_accum, op=dist.ReduceOp.AVG)

                if use_probe_train:
                    if cfg.use_fp16:
                        scaler.step(probe_optimizer)
                        scaler.update()
                    else: probe_optimizer.step()

                if master_process:
                    print(f"epoch:{epoch_eval:2d} | step: {step:4d} | loss_eval_accum: {loss_eval_accum.item():.4f}")

                # if step == 10:
                #     break

            accuracy = correct / total * 100 # unit is %
            if use_ddp:
                dist.all_reduce(accuracy, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"epoch:{epoch_eval} | accuracy: {accuracy.item():.2f}%")

    # TRAIN linear probe
    use_probe_train = True
    print("> Train linear probe classifier <")
    run_probe(dist_sampler, data_loader, use_probe_train)
    save_dict = {
        'probe': probe.state_dict(),
        'probe_optimizer': probe_optimizer.state_dict(),
    }
    if master_process:
        print('save probe checkpoint')
        file = f"checkpoint_end_probe.pt"
        file_dir = "checkpoints"
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, file)
        torch.save(save_dict, file_path)


    # TEST linear probe on val_data
    val_data = load_from_disk(f"{cfg.tiny_data_folder_name}/validation")
    val_data = val_data.with_transform(make_transform)
    val_dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset = val_data,
        num_replicas = world_size,
        rank=rank,
        shuffle=True,
    )

    val_data_loader = DataLoader(
        val_data,
        batch_size=batch_size_per_process,
        collate_fn=mask_collator,
        num_workers=cfg.num_workers,
        sampler=val_dist_sampler,
        drop_last=cfg.drop_last_data,
        pin_memory=cfg.pin_mem,
        persistent_workers=False,
    )

    probe.eval()
    use_probe_train = False
    print("> test of linear probe classifier on val data <")
    with torch.no_grad():
        run_probe(val_dist_sampler, val_data_loader, use_probe_train)

    if use_ddp:
        destroy_process_group()

if __name__ == '__main__':
    main_probe()
