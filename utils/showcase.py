import torch
import random
from layers.Embed import SimplePatch
from einops import rearrange


def show_pos_rec(x, model_visual_outs, patch_len, stride=None):
    """

    :param x: bs * l * c, bs = 1 for test
    :param mask: bs * n
    :param mask_pos: bs * n_vis
    :param index_keep: bs * n_vis
    :param index_keep_pos: bs * n_vis_vis
    :param pred_position: bs * n_vis
    :param label: bs * n_vis
    :return:
    """
    mask, mask_pos, index_keep, index_keep_pos, index_remove_pos, pred_position, label = model_visual_outs
    B, L, C = x.shape
    pred_position = pred_position[0, :]
    label = label[0, :]
    seed = random.randint(0, C-1)
    stride = patch_len if stride is None else stride

    x_patch = x.reshape(B, -1, patch_len)
    patchify = SimplePatch(patch_len, stride)

    x_patch, c_in = patchify(x)  # (bs*c) * n * p

    x_patch, mask, mask_pos, index_keep, index_keep_pos, index_remove_pos, pred_position, label = \
        x_patch[seed], mask[seed], mask_pos[seed], index_keep[seed], index_keep_pos[seed], index_remove_pos[seed], \
            pred_position[seed], label[seed]

    keep = torch.gather(index_keep, dim=0, index=index_keep_pos)
    mask.scatter_(0, keep, True)

    x_true = x_patch * mask.unsqueeze(-1).repeat(1, patch_len)

    patch_values = torch.gather(x_patch, dim=0, index=index_keep.unsqueeze(-1).repeat(1, patch_len))
    patch_values = torch.gather(patch_values, dim=0, index=index_remove_pos.unsqueeze(-1).repeat(1, patch_len))
    label = torch.gather(label, dim=0, index=index_remove_pos)
    pred_position = torch.gather(pred_position, dim=0, index=index_remove_pos)

    pred_results = torch.zeros_like(x_true, dtype=x_true.dtype).\
        scatter_add_(0, pred_position.unsqueeze(-1).repeat(1, patch_len), patch_values)

    dup_count = torch.zeros_like(pred_results, dtype=x_true.dtype)
    dup_count.scatter_add_(0, pred_position.unsqueeze(-1).repeat(1, patch_len), torch.ones_like(x_true))
    dup_count[dup_count == 0] = 1
    pred_results /= dup_count

    wrong_patch = patch_values[label != pred_position]
    wrong_index = pred_position[label != pred_position]
    true_index = label[label != pred_position]

    outs = {'original': x_patch.reshape(-1).tolist(),
            'true': x_true.reshape(-1).tolist(),
            'pred': pred_results.reshape(-1).tolist(),
            'wrong_patch': wrong_patch.tolist(),
            'wrong_index': wrong_index.tolist(),
            'true_index': true_index.tolist()
            }

    return outs


def PatchTST_showcase(visual_out, batch_x, patch_len):
    rec = visual_out['rec']
    mask = visual_out['mask']
    _, L, C = rec.shape
    seed = random.randint(0, C-1)

    rec = rec[0, :, seed]
    mask = mask[0, :, seed]
    batch_x = batch_x[0, L:, seed]
    mask_x = batch_x * ~mask

    visual_outs = {'rec': rec.cpu().tolist(),
                   'true': batch_x.cpu().tolist(),
                   'mask': mask_x.cpu().tolist(),
                   'patch_len': patch_len}

    return visual_outs


def dm_showcase(visual_out, batch_x, patch_len):
    rec = visual_out['rec']
    mask = visual_out['mask'].bool()

    rec = rearrange(rec, 'b l c -> b c l')
    mask = rearrange(mask, 'b l c -> b c l')
    batch_x = rearrange(batch_x, 'b l c -> b c l')

    _, C, L = rec.shape
    seed = random.randint(0, C-1)

    rec = rec[0, :, :]
    mask = mask[0, :, :]

    x_start = batch_x.shape[2] - L
    batch_x = batch_x[0, :, x_start:]
    # print(batch_x.shape)

    discard = batch_x * mask  # discard patches

    if 'mask2' in visual_out.keys():
        mask2 = visual_out['mask2'].bool()
        mask2 = rearrange(mask2, 'b l c -> b c l')

        mask2 = mask2[0, :, :]
        rec = rec * mask2

        vis = batch_x * ~mask * ~mask2  # finally visible patches

        visual_outs = {'rec': rec.cpu().tolist(),
                       'true': batch_x.cpu().tolist(),
                       'discard': discard.cpu().tolist(),
                       'vis': vis.cpu().tolist(),
                       'patch_len': patch_len}
    else:
        vis = batch_x * ~mask
        rec = rec * mask
        visual_outs = {'rec': rec.cpu().tolist(),
                       'true': batch_x.cpu().tolist(),
                       'vis': vis.cpu().tolist(),
                       'patch_len': patch_len}

    return visual_outs


if __name__ == '__main__':
    src = torch.arange(1, 31).reshape((2, 5, 3))
    print(src)
    index = torch.tensor([[0, 1, 2], [0, 1, 4]]).unsqueeze(-1).repeat(1, 1, 3)
    print(index)
    results = torch.zeros(3, 5, 3, dtype=src.dtype).scatter_(1, index, src)

    print(results)

    # values = torch.arange(24).reshape(2, 3, 4)
    # pred_pos = torch.tensor([[2, 3, 1], [2, 1, 0]])

