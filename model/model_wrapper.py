import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as opt
from torchmetrics import MeanSquaredError, MeanAbsoluteError

from model.gru import gru
from model.conv import conv_mixer
from model.mlp import mlp
from model.informer import transformer
from helpers.tools import binary_metrics, custom_corr_regularization, weighted_mse


class Architecture_PL(pl.LightningModule):
    def __init__(
        self,
        n_vars=3,
        max_lags=3,
        trans_max_ts_length=600,
        mlp_max_ts_length=600,
        model_type="bidirectional",
        corr_input=True,
        loss_type="ce",
        val_metric="ME",
        regression_head=False,
        link_thresholds=[0.25, 0.5, 0.75],
        corr_regularization=False,
        soft_adapt=False,
        distinguish_mode=False,
        full_representation_mode=False,
        optimizer_lr=1e-4,
        weight_decay=0.01,
        # scheduler_factor = 0.1,
        # betas = 0.9, i leave that out for now
        # transformer specifics
        d_model=32,
        n_heads=2,
        num_encoder_layers=2,
        d_ff=128,
        dropout=0.05,
        distil=True,
        # gru specifics
        gruU_hidden_size1=10,
        gruU_hidden_size2=10,
        gruU_hidden_size3=10,
        gruU_num_layers=10,
        gruB_hidden_size1=10,
        gruB_hidden_size2=10,
        gruB_hidden_size3=10,
        gruB_num_layers=10,
        # convMixer specifics
        convM_dim=512,
        convM_depth=5,
        # kernel_size = 12, I derive this from the max_lags
        # patch_size = 12,
        convM_hidden_size1=100,
        conv1D=False,
        # mlp specifics
        mlp_hidden_size1=150,
        mlp_hidden_size2=100,
        mlp_hidden_size3=50,
        mlp_hidden_size4=50,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.n_vars = n_vars
        self.max_lags = max_lags

        self.loss_type = loss_type
        self.val_metric = val_metric
        self.optimizer_lr = optimizer_lr
        # self.betas = betas
        # self.scheduler_factor = scheduler_factor
        self.regression_head = regression_head
        self.corr_input = corr_input
        self.weight_decay = weight_decay
        self.link_thresholds = link_thresholds
        self.corr_regularization = corr_regularization
        self.trans_max_ts_length = trans_max_ts_length
        self.mlp_max_ts_length = mlp_max_ts_length
        self.loss_term_scaling = torch.Tensor([2, 0.25, 0.25])
        self.full_representation_mode = full_representation_mode
        self.loss_scaling = {}
        self.distinguish_mode = distinguish_mode
        if self.distinguish_mode:
            self.regression_head = True

        # specific for the model type not used all the time
        self.model_type = model_type

        if self.model_type == "bidirectional":
            self.model = gru(
                max_lags=self.max_lags,
                n_vars=self.n_vars,
                hidden_size1=gruB_hidden_size1,
                hidden_size2=gruB_hidden_size2,
                hidden_size3=gruB_hidden_size3,
                num_layers=gruB_num_layers,
                corr_input=self.corr_input,
                regression_head=self.regression_head,
                direction=self.model_type,
            )

        elif self.model_type == "unidirectional":
            self.model = gru(
                max_lags=self.max_lags,
                n_vars=self.n_vars,
                hidden_size1=gruU_hidden_size1,
                hidden_size2=gruU_hidden_size2,
                hidden_size3=gruU_hidden_size3,
                num_layers=gruU_num_layers,
                corr_input=self.corr_input,
                regression_head=self.regression_head,
                direction=self.model_type,
            )

        elif self.model_type == "convM":
            self.model = conv_mixer(
                convM_dim=convM_dim,
                convM_depth=convM_depth,
                corr_input=self.corr_input,
                kernel_size=self.max_lags * 2 + 1,
                patch_size=self.max_lags * 2 + 1,
                hidden_size1=convM_hidden_size1,
                n_vars=self.n_vars,
                max_lags=self.max_lags,
                regression_head=self.regression_head,
                conv1D=conv1D,
                full_representation_mode=self.full_representation_mode,
            )
        elif self.model_type == "mlp":
            self.model = mlp(
                max_ts_length=self.mlp_max_ts_length,
                max_lags=self.max_lags,
                n_vars=self.n_vars,
                corr_input=self.corr_input,
                hidden_size1=mlp_hidden_size1,
                hidden_size2=mlp_hidden_size2,
                hidden_size3=mlp_hidden_size3,
                hidden_size4=mlp_hidden_size4,
                regression_head=self.regression_head,
                full_representation_mode=self.full_representation_mode,
            )

        elif self.model_type == "transformer":
            self.model = transformer(
                n_vars=n_vars,
                d_model=d_model,
                max_lags=max_lags,
                n_heads=n_heads,
                num_encoder_layers=num_encoder_layers,
                d_ff=d_ff,
                dropout=dropout,
                distil=distil,
                max_length=trans_max_ts_length,
                regression_head=self.regression_head,
                corr_input=corr_input,
            )
        else:
            print("MODEL TYPE NOT KNOWN!")

        self.regression_loss = self.regression_loss_init()
        self.classifier_loss = self.classifier_loss_init()
        self.mse, self.mae = self.val_metrics_init()
        self.weights = torch.Tensor([1, 1, 1])

    def val_metrics_init(self):
        # TODO make this more flexible
        if self.val_metric == "ME":
            return MeanSquaredError(), MeanAbsoluteError()
        else:
            print("NOT IMPLEMENTED")

    def regression_loss_init(self):
        if self.distinguish_mode:
            return nn.BCEWithLogitsLoss()
        if self.regression_head:
            return nn.MSELoss()
        else:
            return None

    def classifier_loss_init(self):
        if self.loss_type == "mse":
            print("init with MSE")
            return nn.MSELoss()
        if self.loss_type == "wmse":
            print("init with WMSE")
            return weighted_mse()
        if self.loss_type == "mae":
            print("init with mae")
            return nn.L1Loss()

        elif self.loss_type == "bce":
            return nn.BCEWithLogitsLoss()
        else:
            return None

    def calc_log_F1_metrics(self, y_class, lab_class, name="no_name"):
        out_d = {}
        for thresh in self.link_thresholds:
            tp, fp, tn, fn = binary_metrics(torch.sigmoid(y_class), lab_class, thresh)
            f1 = tp / (tp + 0.5 * (fp + fn))
            f1 = f1 if not torch.isnan(f1) else 0
            out_d["tp_" + str(thresh) + "_" + name] = tp
            out_d["fp_" + str(thresh) + "_" + name] = fp
            out_d["tn_" + str(thresh) + "_" + name] = tn
            out_d["fn_" + str(thresh) + "_" + name] = fn
            out_d["f1_" + str(thresh) + "_" + name] = f1
            self.log_dict(out_d, sync_dist=True)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        # Change input representation for the MLP.

        y_ = self.model(inputs)
        if self.distinguish_mode:
            reg_loss = self.regression_loss(y_[1], labels)
            class_loss = torch.zeros((1, 1), device=reg_loss.device) + 1e-10
            corr_loss = torch.zeros((1, 1), device=reg_loss.device) + 1e-10

        else:
            if self.regression_head:
                y_class, y_regression = y_
                lab_class, lab_regression = labels
                class_loss = self.classifier_loss(y_class, lab_class)
                reg_loss = self.regression_loss(y_regression, lab_regression)

            else:
                y_class = y_
                lab_class = labels
                class_loss = self.classifier_loss(y_class, lab_class)
                reg_loss = torch.zeros((1, 1), device=class_loss.device) + 1e-10

            if self.corr_regularization:
                raw_data = inputs[0] if self.corr_input else inputs
                corr_loss = custom_corr_regularization(torch.sigmoid(y_class), raw_data)
            else:
                corr_loss = torch.zeros((1, 1), device=class_loss.device) + 1e-10

         
            self.calc_log_F1_metrics(y_class, lab_class, name="train")
            self.log("tr_output_mean", y_class.mean(), sync_dist=True, prog_bar=True)

        if self.corr_regularization:
            self.log(
                "tr_cor_loss",
                corr_loss.type("torch.DoubleTensor"),
                sync_dist=True,
                prog_bar=True,
            )
        if self.regression_head:
            self.log(
                "tr_reg_loss",
                reg_loss.type("torch.DoubleTensor"),
                sync_dist=True,
                prog_bar=True,
            )
        self.log(
            "tr_class_loss",
            class_loss,
            sync_dist=True,
            prog_bar=True,
        )

        loss = (
            corr_loss * self.loss_term_scaling[2]
            + reg_loss * self.loss_term_scaling[1]
            + class_loss * self.loss_term_scaling[0]
        )
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def non_train_step(self, batch, name="no_name"):
        inputs, labels = batch

        y_ = self.model(inputs)
        if self.distinguish_mode:
            reg_loss = self.regression_loss(y_[1], labels)
            class_loss = torch.zeros((1, 1), device=reg_loss.device) + 1e-10
            corr_loss = torch.zeros((1, 1), device=reg_loss.device) + 1e-10
            self.log(
                name + "_loss",
                reg_loss * self.loss_term_scaling[2],
                sync_dist=True,
                prog_bar=True,
            )

        else:
            if self.regression_head:
                y_1 = y_[0]
                l1 = labels[0]
                l2 = labels[1]
                y_2 = y_[1]
                reg_loss = self.regression_loss(y_2, l2)
                self.log(
                    name + "_reg_loss",
                    reg_loss * self.loss_term_scaling[2],
                    sync_dist=True,
                    prog_bar=True,
                )

            else:
                y_1 = y_
                l1 = labels
                reg_loss = 0

            if self.corr_regularization:
                raw_data = inputs[0] if self.corr_input else inputs
                corr_loss = custom_corr_regularization(torch.sigmoid(y_1), raw_data)
                self.log(
                    name + "_corr_loss",
                    corr_loss * self.loss_term_scaling[1],
                    sync_dist=True,
                    prog_bar=True,
                )
            else:
                corr_loss = 0

            class_loss = self.classifier_loss(y_1, l1)
            self.log(
                name + "_class_loss",
                class_loss * self.loss_term_scaling[0],
                sync_dist=True,
                prog_bar=True,
            )

            loss = (
                class_loss * self.loss_term_scaling[0]
                + corr_loss * self.loss_term_scaling[1]
                + reg_loss * self.loss_term_scaling[2]
            )
            mse = self.mse(torch.sigmoid(y_1), l1)

            self.calc_log_F1_metrics(y_1, l1, name=name)
            self.log(name + "_MSE", mse, sync_dist=True, prog_bar=True)
            self.log(name + "_output_mean", y_1.mean(), sync_dist=True, prog_bar=True)
            self.log(name + "_loss", loss, sync_dist=True, prog_bar=True)

    def validation_step(self, batch, _):
        self.non_train_step(batch, name="val")

    def test_step(self, batch, _):
        self.non_train_step(batch, name="test")

    # if we want norms at some point
    # def on_before_optimizer_step(self, optimizer):
    # norms1 = grad_norm(self.model.rnn, norm_type=2)
    # self.log_dict(norms1)

    def configure_optimizers(self):
        optim = opt.AdamW(
            self.model.parameters(),
            lr=self.optimizer_lr,  # betas= self.betas,
            weight_decay=self.weight_decay,
        )
        schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.2)

        return [optim], [{"scheduler": schedule, "monitor": "train_loss"}]
