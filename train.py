import hydra
import lightning.pytorch as pl
import utils
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor,RichModelSummary,RichProgressBar, ModelCheckpoint

@utils.task_wrapper
def train(cfg: DictConfig) -> tuple[dict, dict]:
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)

    
    data_module: LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    logger: TensorBoardLogger = hydra.utils.instantiate(cfg.tensorboard)
    logger.log_hyperparams(cfg)


    checkpoint_callback = ModelCheckpoint(dirpath=cfg.tensorboard.save_dir, save_top_k=1, monitor="val_loss", save_last=True)

    # Callbacks
    es = EarlyStopping(monitor='val_loss',   # not quite sure what is the best here... 
                        patience=cfg.early_stopping.patience,
                        min_delta=cfg.early_stopping.min_delta, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger,strategy='ddp_find_unused_parameters_true',
                                                callbacks=[es,
                                                            lr_monitor,
                                                            checkpoint_callback,
                                                            RichModelSummary(),
                                                            RichProgressBar()], devices=-1)

    object_dict = {"cfg": cfg, "datamodule": data_module, "model": model, "logger": logger, "trainer": trainer}    

    trainer.fit(model=model, datamodule=data_module)
    train_metrics = trainer.callback_metrics
    
    trainer.test(model=model, datamodule=data_module)
    test_metrics = trainer.callback_metrics
    
    return {**train_metrics, **test_metrics}, object_dict


@hydra.main(version_base="1.3", config_path="config", config_name="train.yaml")
def main(cfg: DictConfig):
    utils.extras(cfg)
    metric_dict, _ = train(cfg)
    utils.test_dict_to_csv(metric_dict,cfg)
    return utils.get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))
if __name__ == '__main__':
    main()