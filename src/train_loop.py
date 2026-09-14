import os
import glob
import concurrent.futures
import multiprocessing as mp

import torch
import torch.nn.functional as F

from utils.keras_to_torch import load_champion, save_checkpoint
from model.tfrecord_dataset import make_dataloader
from model.torch_model import export_onnx, freeze_batchnorm

OUTPUT_DIR = "model/tfrecords/self_gen"
MODEL_DIR = "model/model_iteration"
SUPERVISED_DIR = "model/tfrecords"
NUM_WORKERS = 16

POSITIONS_PER_WORKER = 50_000
POSITIONS_PER_FILE = 50_000
MAX_BUFFER_FILES = 48
BATCH_SIZE = 256

EPOCHS = 1
LEARNING_RATE = 1e-5

SUPERVISED_WEIGHT = 0.10
CHAMPION = 26  # WDL softmax champion; tanh V23.keras will not load

DATALOADER_WORKERS = 2
LOG_EVERY = 50


def prune_replay_buffer(buffer_dir, max_files):
    files = glob.glob(os.path.join(buffer_dir, "*.tfrecord"))
    files.sort(key=os.path.getmtime)

    if len(files) > max_files:
        files_to_delete = len(files) - max_files
        print(f"Replay buffer exceeded {max_files} files. Pruning {files_to_delete} old files...")
        for i in range(files_to_delete):
            try:
                os.remove(files[i])
                print(f"  [Deleted] {os.path.basename(files[i])}")
            except Exception as e:
                print(f"  [Error] Failed to delete {files[i]}: {e}")
    else:
        print(f"Replay buffer holds {len(files)}/{max_files} files. No pruning necessary.")


def _wdl_ce(pred, target):
    """CategoricalCrossentropy(from_logits=False) on softmax WDL."""
    return -(target * pred.clamp_min(1e-7).log()).sum(dim=1).mean()


def _policy_ce(logits, target):
    """CategoricalCrossentropy(from_logits=True) on soft 4672-way policy."""
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def train_current_model(iteration, buffer_dir):
    current_pt = os.path.join(MODEL_DIR, f"V{CHAMPION}.pt")
    new_pt = os.path.join(MODEL_DIR, f"V{iteration}.pt")
    new_onnx = os.path.join(MODEL_DIR, f"V{iteration}.onnx")

    print(f"Loading previous model: {current_pt} (or V{CHAMPION}.keras)")

    self_play_files = glob.glob(os.path.join(buffer_dir, "*.tfrecord"))
    steps_per_epoch = max(1, (len(self_play_files) * POSITIONS_PER_FILE) // BATCH_SIZE)

    try:
        model = load_champion(MODEL_DIR, CHAMPION)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load V{CHAMPION}. Did you run the supervised bootstrap? Error: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    model.to(device)
    freeze_batchnorm(model)
    model.train()

    print("Building dataset pipeline from current replay buffer...")
    loader = make_dataloader(
        buffer_dir,
        BATCH_SIZE,
        supervised_dir=SUPERVISED_DIR,
        supervised_weight=SUPERVISED_WEIGHT,
        num_workers=DATALOADER_WORKERS,
    )

    print(f"Training V{iteration} for {EPOCHS} Epoch over the entire Replay Buffer...")
    total_steps = max(1, EPOCHS * steps_per_epoch)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
        eps=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=LEARNING_RATE / 3,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    data_iter = iter(loader)
    try:
        for epoch in range(EPOCHS):
            running = running_v = running_p = 0.0
            logged = 0
            for step in range(steps_per_epoch):
                board, extra, wdl, policy = next(data_iter)
                board = board.to(device, non_blocking=True)
                extra = extra.to(device, non_blocking=True)
                wdl = wdl.to(device, non_blocking=True)
                policy = policy.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    pred_wdl, logits = model(board, extra)
                pred_wdl = pred_wdl.float()
                logits = logits.float()
                v_loss = _wdl_ce(pred_wdl, wdl)
                p_loss = _policy_ce(logits, policy)
                loss = v_loss + p_loss + model.l2_loss()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running += float(loss.detach())
                running_v += float(v_loss.detach())
                running_p += float(p_loss.detach())
                logged += 1
                if logged == LOG_EVERY or step + 1 == steps_per_epoch:
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"  epoch {epoch + 1}/{EPOCHS} step {step + 1}/{steps_per_epoch} "
                        f"loss={running / logged:.4f} v={running_v / logged:.4f} "
                        f"p={running_p / logged:.4f} lr={lr:.2e}"
                    )
                    running = running_v = running_p = 0.0
                    logged = 0
    finally:
        del data_iter
        del loader

    print(f"Saving new generation model: {new_pt}")
    model.to("cpu")
    save_checkpoint(model, new_pt)
    print(f"Exporting ONNX: {new_onnx}")
    export_onnx(model, new_onnx)
    print(f"Success! Optimized ONNX model saved to {new_onnx}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main_orchestrator():
    global SUPERVISED_WEIGHT, CHAMPION

    from self_play import generate_self_play_data
    from arena import run_tournament

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    iteration = 22
    global_batch_counter = 22

    while True:
        print(f"\n========================================================")
        print(f" STARTING ALPHA-ZERO PIPELINE: ITERATION {iteration}")
        print(f"========================================================")

        print("\n>>> STATE 1: GENERATING SELF-PLAY DATA")

        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for worker_id in range(NUM_WORKERS):
                safe_start_batch = global_batch_counter + (worker_id * 1000)
                future = executor.submit(
                    generate_self_play_data,
                    POSITIONS_PER_WORKER,
                    POSITIONS_PER_FILE,
                    OUTPUT_DIR,
                    safe_start_batch,
                    worker_id,
                    CHAMPION,
                    iteration - 1
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"CRITICAL: A worker crashed during generation: {e}")

        global_batch_counter += 1

        print("\n>>> STATE 2: PRUNING REPLAY BUFFER")
        prune_replay_buffer(OUTPUT_DIR, MAX_BUFFER_FILES)

        print("\n>>> STATE 3: TRAINING NEURAL NETWORK")
        train_current_model(iteration, OUTPUT_DIR)

        print("\n>>> STATE 4: RUNNING ARENA")
        promoted, stats = run_tournament(CHAMPION, iteration)

        if promoted:
            CHAMPION = iteration
            print(f"NEW CHAMPION IS V{iteration}")

        with open("model/logs/arena_logs.txt", "a") as f:
            f.write(f"{stats['champion_path']} vs {stats['challenger_path']}:\n")
            f.write(f"\tScore: {stats['champion_points']} - {stats['challenger_points']}\n")
            f.write(f"\tW-L-D: {stats['champion_wins']}-{stats['challenger_wins']}-{stats['draws']}\n")
            if promoted:
                f.write("PROMOTED\n\n")
            else:
                f.write("NOT PROMOTED\n\n")

        SUPERVISED_WEIGHT = max(0.10, SUPERVISED_WEIGHT / 1.50)

        print(f">>> Iteration {iteration} complete. Loop restarting...\n")
        iteration += 1


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main_orchestrator()
