"""
SouL Flame — save / load checkpoints
====================================
STATUS: skeleton.

WHAT TO SAVE
    model.state_dict(), optimizer.state_dict(), cfg, step, epoch, seed

TEST TO WRITE YOURSELF
    save -> load -> same logits for the same input (eval mode)
"""
import torch


def save_checkpoint(path: str, model, optimizer, cfg: dict, step: int, epoch: int):
    """torch.save a dict with everything needed to resume"""
    # torch.save({"model_state_dict": model.state_dict(),
    #             "optimizer_state_dict": optimizer.state_dict(),
    #             "cfg": cfg, "step": step, "epoch": epoch}, path)
    # save the OPTIMIZER too !! adam keeps momentum. without it, resuming = a small restart
    # and save the cfg, otherwise in 2 weeks u wont know which config those weights belong to
    pass


def load_checkpoint(path: str, model, optimizer=None, device="cpu") -> dict:
    """torch.load -> load_state_dict (model + optimizer) -> return metadata"""
    # 1. ckpt = torch.load(path, map_location=device)
    #    map_location -> lets u load a gpu checkpoint on a cpu machine
    # 2. model.load_state_dict(ckpt["model_state_dict"])
    # 3. if optimizer is not None -> load its state too
    # 4. return the rest (cfg, step, epoch)
    # the model must be BUILT with the same cfg before loading, load_state_dict doesnt create layers
    pass
