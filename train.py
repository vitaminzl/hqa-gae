import argparse
import logging
import warnings
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, NeighborLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from hqa_gae.utils import parse_yaml, set_random_seed, DataUtil, get_time, get_default_config
from hqa_gae.models import GAE, create_gae
from hqa_gae.utils.test import test_svm_classify, test_link_prediction

def pretrain(train_dataset, config, device, log_name="logs", valid_dataset=None, test_dataset=None):
    pylogger.info(" Pretrain Task Start ".center(100, "#"))

    tb_logger = TensorBoardLogger(
        save_dir="logs",  # 指定日志文件保存的目录
        name=log_name, 
        version=f"{args.dataset}_{task_name}_{get_time()}"
    )


    if config.get("num_neighbors"):
        train_dataloader = NeighborLoader(train_dataset[0], batch_size=config.batch_size, 
                                          num_neighbors=config.num_neighbors,
                                          directed=False,
                                          num_workers=config.num_workers)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers)

    if config.get("num_neighbors") and args.task != "lp":
        valid_dataloader = NeighborLoader(valid_dataset[0], batch_size=config.batch_size, 
                                          num_neighbors=config.num_neighbors,
                                          directed=False,
                                          num_workers=config.num_workers)  if valid_dataset else None
        test_dataloader = NeighborLoader(test_dataset[0], batch_size=config.batch_size, 
                                         num_neighbors=config.num_neighbors,
                                          directed=False,
                                         num_workers=config.num_workers)  if test_dataset else None     
    else:
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=config.num_workers)  if valid_dataset else None
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers)  if test_dataset else None


    hqa = create_gae(config, train_dataset[0])

    vq_train_model = GAE(
    hqa, 
    all_edge_index=train_dataset[0].edge_index if config.get("num_neighbors") else None,
    optim=config.optim)
    # split_edge=train_data.split_edge)
    vq_train_model.save_hyperparameters(config.to_dict())
    if args.task == "lp":
        early_stopping_callback = EarlyStopping(
            monitor="valid_AUC", 
            mode="max", 
            patience=config.patience)  
    else:   
        early_stopping_callback = EarlyStopping(
            monitor="train_loss", 
            mode="min", 
            patience=config.patience)
        
    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", None), 
        max_steps=config.get("max_steps", -1),
        log_every_n_steps=1,
        val_check_interval=config.get("val_freq", 1) if config.get("max_steps") else None,
        check_val_every_n_epoch=config.get("val_freq", 1) if config.get("max_epochs") else None,
        callbacks=[
            early_stopping_callback,
        ],
        logger=tb_logger,
        accelerator=device, 
        devices=[device_id] if args.device == "gpu" else "auto")
    
    val_dataloaders=[valid_dataloader, test_dataloader]
    filter_none = lambda x: [i for i in x if i is not None]
    val_dataloaders = filter_none(val_dataloaders)

    trainer.fit(vq_train_model, train_dataloader, 
                val_dataloaders=val_dataloaders)

    if args.task == "lp":
        hqa = vq_train_model.best_model
    else:
        hqa = vq_train_model.model
    pylogger.info(" Pretrain Task End ".center(100, "#"))

    return hqa



def main(config):
    dataset = DataUtil.get_dataset("datasets", args.dataset)
    graph_cls = len(dataset) > 1
    if args.task == "lp":
        train_data, valid_data, test_data = T.RandomLinkSplit(num_val=0.05, num_test=0.1,
                                                    is_undirected=True,
                                                    split_labels=True,
                                                    add_negative_train_samples=True)(dataset[0])
        dataset._data = train_data
        train_dataset, valid_dataset, test_dataset = [train_data], [valid_data], [test_data]
    else:
        train_dataset, valid_dataset, test_dataset = dataset, None, None

    hqa = pretrain(
        train_dataset=train_dataset, 
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        config=config, 
        device=config.device,
        log_name=f"{args.dataset}")

    data_loader = DataLoader(dataset if test_dataset is None else test_dataset, batch_size=2**12, shuffle=False, num_workers=4)
    emb = []
    for batch_data in data_loader:
        z, _ = hqa.get_embedding(batch_data.x, batch_data.edge_index, indices=True)
        z = z.detach()
        emb.append(z)
    emb = torch.concat(emb, dim=0)

    test_restult = downstream_test(dataset, config, emb, 
                                   edge_test_data=test_dataset, graph_cls=graph_cls)

    return test_restult


def downstream_test(dataset, config, emb, edge_test_data=None, graph_cls=False):
    pylogger.info(" Dowmstream Task Start ".center(100, "#"))

    result = {}
    
    if args.task == "lp":
        device = "cpu" if args.device == "cpu" else f"cuda:{device_id}"
        emb = emb.to(device)
        result = test_link_prediction(emb,
                                        pos_edges=edge_test_data[0].pos_edge_label_index, 
                                        neg_edges=edge_test_data[0].neg_edge_label_index,
                                        batch_size=65536, logger=pylogger)

    elif args.task == "nc":
        if config.l2_norm:
            emb = F.normalize(emb.clone(), p=2, dim=-1)
        result = test_svm_classify(emb, dataset.y, 
                                    pylogger, using_cuml=args.using_cuml)
            
        pylogger.info(" Dowmstream Task End ".center(100, "#"))
    
    return {**result}
   
if __name__ == "__main__":
    pylogger = logging.getLogger("pytorch_lightning")
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser("args for graphttt")
    parser.add_argument("--dataset", type=str, required=True, help="cora, citeseer, pubmed, ...")
    parser.add_argument("--config", type=str, default="configs/", help="the file location of config")
    parser.add_argument("--task", type=str, default="nc", help="nc or lp, meaning node classification or link prediction")
    parser.add_argument("--device", type=str, default="gpu", required=False, help="device")
    parser.add_argument("--device_id", type=int, default=0, required=False, help="device_id")
    parser.add_argument("--seed", type=int, default=2024, required=False, help="random seed")
    parser.add_argument("--num_runs", type=int, default=1, required=False, help="num runs")
    parser.add_argument("--using_cuml", action="store_true", help="using cuml or not")

    args = parser.parse_args()
    device_id = args.device_id
    config_df = get_default_config()
    config = parse_yaml(osp.join(args.config, args.dataset + ".yaml"))
    config = config.lp if args.task == "lp" else config.nc
    config.update(config_df.to_dict())
    config.update({"device": args.device})
    dataset_name = args.dataset

    set_random_seed(args.seed)

    if args.using_cuml:
        import rmm
        rmm.reinitialize(devices=[args.device_id])

    task_name = args.task

    all_results = {}

    for i in range(args.num_runs):
        pylogger.info(f" Num Runs {i+1}/{args.num_runs} for {args.dataset} ".center(100, "="))
        all_results = main(config)


    # if args.task == "lp":
    #     pylogger.info(f"All runs Link Prediction AUC: {np.mean(all_test_auc):.4f} +- {np.std(all_test_auc):.4f}")
    #     pylogger.info(f"All runs Link Prediction AP: {np.mean(all_test_ap):.4f} +- {np.std(all_test_ap):.4f}")
    # elif args.task == "nc":
    #     # pylogger.info(f"All runs Linear Prob Test {criterion}: {np.mean(all_test_accs):.4f} +- {np.std(all_test_accs):.4f}")
    #     pylogger.info(f"All runs SVM Test accs: {np.mean(all_svm_accs):.4f} +- {np.std(all_svm_accs):.4f}")
    #     pylogger.info(f"All runs SVM Test Micro F1: {np.mean(all_svm_f1_mic):.4f} +- {np.std(all_svm_f1_mic):.4f}")
    #     pylogger.info(f"All runs SVM Test Macro F1: {np.mean(all_svm_f1_mac):.4f} +- {np.std(all_svm_f1_mac):.4f}")
