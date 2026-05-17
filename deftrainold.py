def train_weights():
    TOTAL_EPOCHS = 1600
    accumulation_steps = 4
    START_EPOCH = 0

    local_checkpoint_path = "/content/rethined_checkpoint.pth"
    local_best_checkpoint_path = "/content/rethined_checkpoint_best.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dirs = [
        "/content/rethined/datasets/DF8K-Inpainting/masks/test/cafhq/",
        "/content/rethined/datasets/DF8K-Inpainting/masks/test/div2k/",
        "/content/rethined/datasets/DF2K",
        "/content/rethined/datasets/CAF/images"


    ]

    dataset = HighResInpaintingDataset(image_dirs)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=4)

    config = {
        'coarse_model': {'class': 'MobileOneCoarse', 'parameters': {'variant': 's4'}},
        'generator': {
            'generator_class': 'PatchInpainting',
            'params': {
                'kernel_size': 8, 'nheads': 1, 'stem_out_stride': 1, 'stem_out_channels': 3,
                'merge_mode': 'all', 'image_size': 512, 'embed_dim': 576, 'use_qpos': None,
                'use_kpos': None, 'dropout': 0.1, 'feature_i': 2, 'concat_features': True,
                'final_conv': True, 'feature_dim': 896, 'attention_type': 'MultiHeadAttention',
                'compute_v': False, 'use_argmax': False
            }
        }
    }

    model = InpaintingModel(config).to(device)
    attention_upscaler = AttentionUpscaling(model.generator).to(device)

    is_resuming = False

    if os.path.exists(local_checkpoint_path):
        print("Found checkpoint locally. Restoring automatically.")
        checkpoint = torch.load(local_checkpoint_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            START_EPOCH = checkpoint.get('epoch', 0) + 1
            print(f"Resuming training from Epoch {START_EPOCH}")
        else:
            state_dict = checkpoint
            print("Notice: Loaded a raw weights file. Starting epoch count from 0.")

        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

        is_resuming = True
    else:
        print("No previous data found locally. Starting training from scratch (Epoch 0).")

    for param in model.parameters():
        param.requires_grad = True
    model.train()

#learning rate
    current_lr = 5e-6

    torch._dynamo.config.suppress_errors = True
    try:
        model = torch.compile(model, backend = "inductor", options = {"triton.cudagraphs": False})
    except Exception as e:
        print(f"Compile Info:{e}")


    optimizer = AdamW(model.parameters(), lr=current_lr, weight_decay=1e-5)

    if is_resuming and os.path.exists(local_checkpoint_path):
        try:
            if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Successfully restored optimiser parameters from Epoch {START_EPOCH - 1}.")
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                    if 'initial_lr' in param_group:
                        param_group['initial_lr'] = current_lr
                print(f"Update learning rate: {current_lr}")

        except Exception as e:
            print(f"Warning: Could not load optimiser ({e}). Reinitialising.")

    l1_loss_fn = torch.nn.L1Loss()
    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.0)

    epochs_left = TOTAL_EPOCHS - START_EPOCH
    if epochs_left <= 0:
        print("Model has completed the required number of epochs.")
        return

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_left, eta_min=1e-7)

    if is_resuming and os.path.exists(local_checkpoint_path):
        try:
            if isinstance(checkpoint, dict) and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load scheduler or scaler ({e}).")

    model.train()
    if is_resuming:
        model.coarse_model.eval()

    best_loss = float('inf')

    if is_resuming and os.path.exists(local_checkpoint_path):
        if isinstance(checkpoint, dict) and 'best_loss' in checkpoint:
            best_loss = checkpoint['best_loss']
            print(f"Previous Best Loss: {best_loss:.4f}")

    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        epoch_loss = 0.0
        nan_count = 0

        progress_bar = tqdm(enumerate(dataloader),
                            total=len(dataloader),
                            desc=f"Epoch {epoch}/{TOTAL_EPOCHS}")

        optimizer.zero_grad(set_to_none=True)

        for i, (hr_img, hr_mask) in progress_bar:
            torch.compiler.cudagraph_mark_step_begin()
            hr_img, hr_mask = hr_img.to(device, non_blocking=True), hr_mask.to(device, non_blocking=True)

            lr_img = F.interpolate(hr_img, size=512, mode='bilinear', antialias=True)
            lr_mask = F.interpolate(hr_mask, size=512)
            masked_lr_img = lr_img * (1 - lr_mask)

            with torch.amp.autocast('cuda'):
                lr_out, attn_scores, coarse_out = model(masked_lr_img, lr_mask)
                hr_out = attention_upscaler(hr_img, lr_out, attn_scores)

                loss_l1_coarse = l1_loss_fn(coarse_out, lr_img)
                loss_l1_lr = l1_loss_fn(lr_out, lr_img)
                loss_l1_hr = l1_loss_fn(hr_out, hr_img)

                hr_out_for_vgg = F.interpolate(hr_out, size=512, mode='bilinear')
                hr_img_for_vgg = F.interpolate(hr_img, size=512, mode='bilinear')

                loss_perc_lr = perceptual_loss_fn(lr_out, lr_img)
                loss_perc_hr = perceptual_loss_fn(hr_out_for_vgg, hr_img_for_vgg)

                total_loss = (loss_l1_coarse + loss_l1_lr + loss_l1_hr + 0.1 * (loss_perc_lr + loss_perc_hr)) / accumulation_steps

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                nan_count += 1
                if nan_count <= 3:
                    print(f"\nWarning: Detected NaN/Inf Loss at Step {i}. Skipping to protect model.")
                elif nan_count == 4:
                    print(f"\nNotice: Multiple NaNs detected. Muting further warnings for this epoch.")

                optimizer.zero_grad(set_to_none=True)

                if nan_count > 20:
                    raise ValueError("Model is severely corrupted with NaNs. Training halted to prevent overwriting healthy data. Please delete 'rethined_checkpoint.pth' and resume from 'rethined_checkpoint_best.pth'.")
                continue

            scaler.scale(total_loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            display_loss = total_loss.item() * accumulation_steps
            epoch_loss += display_loss

            current_display_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({'loss': f"{display_loss:.4f}", 'lr': f"{current_display_lr:.6e}"})

        if nan_count >= len(dataloader) * 0.9:
            raise ValueError("Almost all steps returned NaN. The current checkpoint weights are corrupted.")

        avg_loss = epoch_loss / max(1, (len(dataloader) - nan_count))
        print(f"\nEpoch {epoch}/{TOTAL_EPOCHS} completed - Average Loss: {avg_loss:.4f}")

        scheduler.step()

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': best_loss
        }

        temp_local_path = local_checkpoint_path + ".tmp"
        torch.save(checkpoint_data, temp_local_path)
        os.replace(temp_local_path, local_checkpoint_path)
        print(f"Successfully saved checkpoint for Epoch {epoch} to local storage.")

        if avg_loss < best_loss:
            print(f"New Record! Loss decreased from {best_loss:.4f} down to {avg_loss:.4f}")
            best_loss = avg_loss

            temp_best_path = local_best_checkpoint_path + ".tmp"
            torch.save(model.state_dict(), temp_best_path)
            os.replace(temp_best_path, local_best_checkpoint_path)

            print("Safely updated best checkpoint file on local storage.")
