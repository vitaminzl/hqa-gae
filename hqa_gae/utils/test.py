import cuml
import copy
import cupy as cp
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import  EarlyStopping
from hqa_gae.utils.optim import Criterion
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score, average_precision_score, silhouette_score
from sklearn.model_selection import  KFold, StratifiedKFold, GridSearchCV
from hqa_gae.utils.optim import build_optimizer

def test_svm_classify(feature, labels, logger=None, using_cuml=True, just_one_fold=False):
    f1_mac = []
    f1_mic = []
    accs = []
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    with tqdm(kf.split(feature), total=1 if just_one_fold else 5, desc="5-fold cross validation") as tqdmEpoch:
        for train_index, test_index in tqdmEpoch:
            train_X, train_y = feature[train_index].cpu().detach().numpy(), labels[train_index].cpu().detach().numpy()
            test_X, test_y = feature[test_index].cpu().detach().numpy(), labels[test_index].cpu().detach().numpy()
            if using_cuml:
                from cuml import svm
                clf = svm.SVC(kernel='rbf', verbose=cuml.common.logger.level_error)
                clf.fit(cp.array(train_X), cp.array(train_y))
            else:
                from sklearn import svm
                clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
                clf.fit(train_X, train_y)
            preds = clf.predict(test_X)
            micro = f1_score(test_y, preds, average='micro')
            macro = f1_score(test_y, preds, average='macro')
            acc = accuracy_score(test_y, preds)

            tqdmEpoch.set_postfix({'f1_micro': f'{micro:.4f}',
                                  'f1_macro': f'{macro:.4f}', 'acc': f'{acc:.4f}'})
            tqdmEpoch.update()
            accs.append(acc)
            f1_mac.append(macro)
            f1_mic.append(micro)
            if just_one_fold:
                break

    f1_mic_mean = np.mean(f1_mic)
    f1_mac_mean = np.mean(f1_mac)
    accs_mean = np.mean(accs)
    if logger is not None:
        logger.info(f'Testing based on svm: f1_micro={f1_mic_mean:.4f}, f1_macro={f1_mac_mean:.4f}, acc={accs_mean:.4f}')
    return {"f1_mic": f1_mic, 
            "f1_mac": f1_mac, 
            "acc": accs}


def test_svm_graph_classify(embeddings, labels, logger=None, using_cuml=True):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    f1_mac = []
    f1_mic = []
    accs = []
    for train_index, test_index in tqdm(kf.split(embeddings, labels), total=10, desc="10-fold cross validation"):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        if using_cuml:
            from cuml import svm
            model = svm.SVC(random_state=42)
            clf = GridSearchCV(model, params)
            clf.fit(cp.array(x_train.cpu().numpy()), cp.array(y_train.cpu().numpy()))
        else:
            from sklearn import svm
            model = svm.SVC(random_state=42)
            clf = GridSearchCV(model, params, n_jobs=4)
            # clf = model
            clf.fit(x_train.cpu().numpy(), y_train.cpu().numpy())

        preds = clf.predict(x_test)
        micro = f1_score(y_test, preds, average="micro")
        macro = f1_score(y_test, preds, average="macro")
        acc = accuracy_score(y_test, preds)
        accs.append(acc)
        f1_mac.append(macro)
        f1_mic.append(micro)
    f1_mic_mean = np.mean(f1_mic)
    f1_mac_mean = np.mean(f1_mac)
    accs_mean = np.mean(accs)
    if logger is not None:
        logger.info(f'Testing based on svm: f1_micro={f1_mic_mean:.4f}, f1_macro={f1_mac_mean:.4f}, acc={accs_mean:.4f}')
    return {"f1_mic": f1_mic, 
            "f1_mac": f1_mac, 
            "acc": accs}

def test_cluster(x, y, logger=None, using_cuml=True):
    ARIs = []
    NMIs = []
    SCs = []
    x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()
    for i in tqdm(range(5), desc="kmeans cluster"):
        if using_cuml:
            from cuml import cluster
            kmeans = cluster.KMeans(n_clusters=(y.max() + 1), random_state=i, verbose=cuml.common.logger.level_error)
            y_pred = kmeans.fit_predict(x)
        else:
            from sklearn import cluster
            kmeans = cluster.KMeans(n_clusters=(y.max() + 1), n_init=20, random_state=i)
            y_pred = kmeans.fit_predict(x)
        NMI = normalized_mutual_info_score(y, y_pred)
        ARI = adjusted_rand_score(y, y_pred)
        SC = silhouette_score(x, y_pred)
        ARIs.append(ARI)
        NMIs.append(NMI)
        SCs.append(SC)
    if logger is not None:
        logger.info(f'Testing based on kmeans: NMI={np.mean(NMIs):.4f}, ARI={np.mean(ARIs):.4f}, SC={np.mean(SCs):.4f}')
        logger.info(f'Testing based on kmeans: max NMI={np.max(NMIs):.4f}, max ARI={np.max(ARIs):.4f}, max SC={np.max(SCs):.4f}')

    return {"ARI": ARIs, "NMI": NMIs}



def evaluate_auc(pred, label):
    auc = roc_auc_score(label, pred)
    ap = average_precision_score(label, pred)
    return {
        "AUC": auc,
        "AP": ap
    }


class DotEdgeDecoder(nn.Module):
    """Simple Dot Product Edge Decoder"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def reset_parameters(self):
        return

    def forward(self, z, edge, sigmoid=True):
        x = z[edge[0]] * z[edge[1]]
        x = x.sum(-1)

        if sigmoid or not self.training:
            x = x.sigmoid()
        else:
            x = x
        
        return x

@torch.no_grad()
def test_link_prediction(embedding: torch.Tensor, pos_edges: torch.Tensor, neg_edges: torch.Tensor, 
                         batch_size: int, logger: logging.Logger | None=None):
    edge_predictor = DotEdgeDecoder().to(embedding.device)
    edge_predictor.eval()
    
    edges = torch.concat([pos_edges, neg_edges], dim=1).to(embedding.device)
    train_preds = []
    for edges in DataLoader(edges.T, batch_size, shuffle=False):
        train_preds += [edge_predictor(embedding, edges.T).squeeze().cpu()]
    pred = torch.cat(train_preds, dim=0)
    label = torch.cat([torch.ones(pos_edges.size(1)), torch.zeros(neg_edges.size(1))], dim=0)

    results = evaluate_auc(pred, label)
    if logger is not None:
        eval_str = f"Link Prediction Test AUC: {results['AUC']:.4f}, AP: {results['AP']:.4f}"
        logger.info(eval_str)

    return results

class LightingTestModelWrapper(pl.LightningModule):
    def __init__(self, model: nn.Module, optim: dict, criterion=None):
        super().__init__()
        self.model = model
        self.best_model_state = model.state_dict()
        self.best_score = 0
        self.optim = optim
        self.criterion = criterion if criterion else Criterion("acc")

    def training_step(self, batch, batch_idx):
        self.model.train()
        x, y = batch
        out = self.model(x)

        loss = F.cross_entropy(out, y.squeeze())
        self.log(f"train_{self.criterion}", loss, prog_bar=True)

        y_pred = out.max(1)[1]
        train_metric = self.evaluate(y_pred=y_pred, y_true=y)
        self.log(f"train_{self.criterion}", train_metric, prog_bar=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        x, y = batch
        out = self.model(x)

        y_pred = out.max(1)[1]
        val_metric = self.evaluate(y_pred=y_pred, y_true=y)
        self.log(f"val_{self.criterion}", val_metric, prog_bar=True)
        if val_metric > self.best_score:
            self.best_score = val_metric
            self.best_model_state = self.model.state_dict()

    def get_best_model(self):
        best_model = copy.deepcopy(self.model)
        best_model.load_state_dict(self.best_model_state)
        return best_model

    def evaluate(self, y_pred, y_true):
        if self.criterion:
            metric = self.criterion(**{"y_true": y_true, "y_pred": y_pred.unsqueeze(1)})
        else:
            metric = y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0]

        return metric

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        model = self.get_best_model()
        model.eval()
        x, y = batch
        out = model(x)

        y_pred = out.max(1)[1]
        test_metric = self.evaluate(y_pred=y_pred, y_true=y)
        self.log(f"test_{self.criterion}", test_metric)

    def configure_optimizers(self):
        optimizer, scheduler = build_optimizer(
            self.optim.optimizer,
            self.optim.scheduler,
            self.model.parameters())
        if scheduler:
            return [optimizer], [scheduler]
        else:
            return optimizer


def linear_classify(emb, y, model, config, device, *, 
               train_mask, val_mask, test_mask, device_id, log_name="logs"):
    from torch.utils.data import DataLoader, TensorDataset
    train_dataloader = DataLoader(TensorDataset(emb[train_mask], y[train_mask]), 
                            num_workers=config.train.num_workers, 
                            batch_size=config.train.batch_size)
    val_dataloader = DataLoader(TensorDataset(emb[val_mask], y[val_mask]), 
                            num_workers=config.train.num_workers, 
                            batch_size=config.train.batch_size)
    
    test_dataloader = DataLoader(TensorDataset(emb[test_mask], y[test_mask]), 
                            num_workers=config.train.num_workers, 
                            batch_size=config.train.batch_size)


    criterion = Criterion(**config.criterion)

    lightning_model = LightingTestModelWrapper(
        model,
        criterion=criterion,
        optim=config.optim,)
    lightning_model.save_hyperparameters(config.to_dict())

    # Setup Pytorch Lighting Callbacks
    early_stopping_callback = EarlyStopping(
        monitor=f"val_{criterion}", 
        mode="max", 
        patience=50)

    
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    trainer = pl.Trainer(max_steps=config.train.get("max_steps", -1),
                         max_epochs=config.train.get("max_epochs", None), 
                        callbacks=[
                            early_stopping_callback,
                        ],
                        val_check_interval=config.train.val_freq if config.train.get("val_freq") and config.train.get("max_steps") else None,
                        check_val_every_n_epoch=config.train.get("val_freq", 1),
                        enable_progress_bar=True,
                        enable_model_summary=False,
                        enable_checkpointing=False,
                        accelerator=device, 
                        logger=False,
                        devices=[device_id])
    trainer.fit(lightning_model, train_dataloader,
                val_dataloaders=val_dataloader)

    # Compute validation and test accuracy
    val_acc = early_stopping_callback.best_score.item()
    test_acc = trainer.test(lightning_model, verbose=False,
                            dataloaders=test_dataloader)[0][f"test_{criterion}"]
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    return val_acc, test_acc, criterion


class TestHead(nn.Module):
    def __init__(self, input_dim, output_dim, pooling="max") -> None:
        super().__init__()
        self.out_proj = nn.Linear(input_dim, output_dim)
        # nn.init.xavier_uniform_(self.out_proj.weight.data)
        # nn.init.zeros_(self.out_proj.bias.data)

    def forward(self, x):
        x = self.out_proj(x)
        return x

def test_linear_classify(emb, dataset, config, device_id, pylogger):
    val_accs, test_accs, criterion = [], [], None
    for num_run in tqdm(range(config.downstream.num_runs), desc=f"{config.downstream.num_runs} runs Linear Prob"):
        model = TestHead(
            input_dim=emb.size(1),
            output_dim=dataset.num_classes, 
            pooling="max")
        config.downstream.update({
            "dataset": config.dataset.to_dict(),
        })
        val_acc, test_acc, criterion = linear_classify(
            emb, 
            dataset.y,
            model, 
            config.downstream, 
            config.device,
            device_id=device_id,
            train_mask=dataset.train_mask,
            val_mask=dataset.val_mask,
            test_mask=dataset.test_mask,
            log_name=f"{config.dataset.name}_logs")
        test_accs.append(test_acc)
        val_accs.append(val_acc)
        str_len = 100
    # pylogger.info("#" * str_len)
    test_accs = test_accs if test_accs else [0]
    criterion = criterion if criterion else "None"
    acc_str = f"Linear Prob Test {criterion}: {np.mean(test_accs).item(): .4f} +- {np.std(test_accs).item(): .4f} "
    pylogger.info(acc_str)
    return {"linear_acc": test_accs,
            "linear_criterion": criterion}

