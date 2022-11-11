"""
    Dump pseudo labels from well-trained SSL models
    and then please refer to
        train_with_tag.py / train_with_tag2.py
"""

import torch
import numpy as np
from tutils import trans_args, trans_init, dump_yaml, tfilename, save_script, print_dict, CSVLogger
import argparse
from models.network_emb_study import UNet_Pretrained
from utils.tester.tester_ssl_debug import Tester
from utils.tester.tester_ssl_multi import Tester as Tester2


def dump_labels_from_ssl(logger, config, model):
    logger.info(f"Oneshot id: {oneshot_id}")
    # Dump labels
    logger.info(f"Save dir: {config_base['runs_dir']}")
    tester = Tester(logger, config, split="Train", default_oneshot_id=oneshot_id,
                    collect_sim=False, upsample="bilinear")
    res = tester.test(model, dump_label=True, oneshot_id=oneshot_id)
    # print(res)
    logger.info(res)

    # Test "Test1+2"
    tester = Tester(logger, config, split="Test1+2", default_oneshot_id=oneshot_id,
                    collect_sim=False, upsample="bilinear")
    res = tester.test(model, oneshot_id=oneshot_id)
    logger.info("Test1+2 info:")
    logger.info(res)
    res['oneshot_id'] = oneshot_id
    csvlogger.record(res)

def dump_labels_from_ssl_multi(logger, config, model):
    logger.info(f"Oneshot id: {oneshot_id}")
    # Dump labels
    logger.info(f"Save dir: {config_base['runs_dir']}")
    tester = Tester2(logger, config, split="Train", default_oneshot_id=oneshot_id,
                    collect_sim=False, upsample="bilinear")

    res, record_list = tester.test(model, dump_label=True, oneshot_ids=oneshot_id)
    # print(res)
    logger.info(res)

    # Test "Test1+2"
    # tester = Tester2(logger, config, split="Test1+2")
    tester = Tester(logger, config, default_oneshot_id=oneshot_id,
                    collect_sim=False, split="Test1+2", upsample="bilinear")
    res, record_list = tester.test(model, oneshot_ids=oneshot_id)
    logger.info("Test1+2 info:")
    logger.info(res)
    res['oneshot_id'] = oneshot_id
    csvlogger.record(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ssl/ssl.yaml")
    parser.add_argument("--experiment", default="dump_labels")
    # parser.add_argument("--temp", type=int, default=114)
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    print_dict(config)

    csvlogger = CSVLogger(logdir=config['base']['runs_dir'])
    # oneshot_id = args.temp             /home1/quanquan/code/landmark/code/runs/ssl/ssl/debug2/ckpt/best_model_epoch_930.pth
    ckpt = config['training']['ckpt'] = "/home1/quanquan/code/landmark/code/runs/ssl/ssl/collect_sim/ckpt_v/model_best.pth"
    # ckpt = config['training']['ckpt'] = '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/pretrain1/ckpt/best_model_epoch_240.pth'
    # ckpt = config['training']['ckpt'] = '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_3/ckpt/best_model_epoch_280.pth'
    # ckpt = config['training']['ckpt'] = '/home1/quanquan/code/landmark/code/runs/ssl/ssl_mixed/debug2/model_best_epoch_2003.pth'
    # ckpt = config['training']['ckpt'] = '/home1/quanquan/code/landmark/code/runs/ssl/ssl_pos_ip/debug/ckpt_v/model_best.pth'
    # ckpt = config['training']['ckpt'] = '/home1/quanquan/code/landmark/code/runs/ssl/ssl_mixup0/debug2/ckpt_v/model_best.pth'
    # ckpt = config['training']['ckpt'] = '/home1/quanquan/code/landmark/code/runs/ssl/interpolate/collect_sim/ckpt_v/model_best.pth'
    model = UNet_Pretrained(3, emb_len=config['special']['emb_len'], non_local=config['special']['non_local'])
    state_dict = torch.load(ckpt)
    model.load_state_dict(state_dict)
    model.cuda()

    config_base = config['base']

    # oneshot_id = [120, 121, 127, 95, 114]
    # oneshot_id = [27, 89, 140, 126, 132, 120, 121, 127, 95, 114]
    oneshot_id = [114]
    if len(oneshot_id) == 1:
        oneshot_id = oneshot_id[0]
        dump_labels_from_ssl(logger, config, model)
    else:
        dump_labels_from_ssl_multi(logger, config, model)
    # for i in range(150):
    #     oneshot_id = i
    #     runs_dir = tfilename(config_base['base_dir'], config_base['experiment'], "prob_3_id_"+str(oneshot_id))
    #     config_base['runs_dir'] = runs_dir
    #     print()
    #     dump_labels_from_ssl(logger, config, model)

    dump_yaml(logger, config)