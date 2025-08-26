
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy


from model.dinowreg import RetAug , RetinaDINOLightning
from data_pipeline.data_set import UniformTrainDataloader

def train_dino(config):
    # Initialize augmentations
    augmentor = RetAug(img_size=config['img_size'])
    
    # Create DataModule
    dataset_names = ["eyepacs" , "aptos" , "ddr" , "idrid"]
    uniform_data_ld = UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=augmentor,
        batch_size=config['batch_size'],
        num_workers=4,
        sampler=True
    )

    data_ld = uniform_data_ld.get_loader()
    
    # Initialize model
    model = RetinaDINOLightning(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        warmup_epochs=config['warmup_epochs'],
        max_epochs=config['max_epochs'],
        batch_size=config['batch_size']
    )
    
    # Training callbacks
    checkpoint_cb = ModelCheckpoint(
        monitor='train_loss',
        filename='dino-retina-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Trainer configuration
    trainer = Trainer(
        accelerator='gpu',
        devices=config['num_gpus'],
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=config['max_epochs'],
        accumulate_grad_batches=config['accumulate_grad'],
        precision='16-mixed',
        callbacks=[checkpoint_cb, lr_monitor],
        gradient_clip_val=0.3,
        log_every_n_steps=50
    )
    
    trainer.fit(model, data_ld)

# Example configuration
config = {
    'img_size': 512,
    'patch_size': 16,
    'embed_dim': 768,
    'warmup_epochs': 10,
    'max_epochs': 100,
    'batch_size': 128,
    'num_workers': 8,
    'num_gpus': 4,
    'accumulate_grad': 2
}

if __name__ == '__main__':
    train_dino(config)