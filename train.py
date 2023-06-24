from argparse import ArgumentParser
import torch.distributed as dist
import torch
import torch.nn
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.optim.lr_scheduler import StepLR
from Scheduler.Noam import NoamScheduler
from Scheduler.CosineWarmupScheduler import CosineWarmupScheduler
from Scheduler.PlateauScheduler import MetricTracker, PlateauScheduler
from utils.plotting import plot_features_sequence, plot_features
from data.DataModul import SPKDataModul
from Modul.VisionTransformer import vit_s, vit_m
from Modul.Dummy import CreateDummy
from data.speaker_encoder import SpeakerEncoder
from utils import score
from timm.models.layers import trunc_normal_
from utils.Helpers import build_tsne
from sys import platform

class Task(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.speedup:
            print("optimizing speed")
            torch.autograd.set_detect_anomaly(False)
            torch.autograd.profiler.emit_nvtx(False)
            torch.backends.cudnn.benchmark = True

        self.trials = np.loadtxt(self.hparams.trial_path, str)
        self.speaker_encoder = SpeakerEncoder(self.hparams.train_csv_path, self.hparams.valid_csv_path, self.hparams.test_csv_path, self.hparams.train2_csv_path, self.hparams.test)
        n_classes = self.speaker_encoder.speaker_count

        self.pretrain = self.hparams.pretrain
        if self.hparams.modul_name == "VisionTransformer":
            if self.hparams.model_size == "small":
                self.model = vit_s(self.pretrain, n_classes, self.hparams.shuffle_type, self.hparams.second, self.hparams.hop, self.hparams.cls_pos)
            if self.hparams.model_size == "medium":
                self.model = vit_m(self.pretrain, n_classes, self.hparams.shuffle_type, self.hparams.second, self.hparams.hop, self.hparams.cls_pos)
            return
        elif self.hparams.modul_name == "Dummy":
            self.model = CreateDummy()
            return
        else:
            raise ValueError("module_name name error")

    def forward(self, x):
        #used during inference, make sure to call model.eval() & with torch.no_grad():
        a = 1
        #feature = self.mel_trans(x)
        #embedding = self.model(feature)
        #return embedding

    def getLosses(self, batch, batch_idx, train=False, test=False):
        train_type = "train" if train else "valid"
        if self.hparams.modul_name == "VisionTransformer":
            loss, x, mask, tgt = self.model(batch, train=train, test=test)
            if(batch_idx == 0 and self.pretrain and train):
                plot_features(self.model.unpatchify(x)[0],title='patches')
                plot_features(self.model.unpatchify(tgt)[0],title='patches_org', name='original_')
        elif self.hparams.modul_name == "Dummy":
            loss, embedding, mel = self.model(batch)
        elif self.hparams.modul_name == "BasicTransformer":
            loss, embedding, mel = self.model(batch)

        if type(loss) is tuple:
            self.log(train_type+'_loss', loss[0], prog_bar=True, sync_dist=True)
            self.log(train_type+'_acc', loss[1], prog_bar=True, sync_dist=True)
            return loss[0]
        else:
            self.log(train_type+'_loss', loss, prog_bar=True, sync_dist=True)
            return loss

    def on_train_start(self):
        #https://github.com/Lightning-AI/lightning/issues/12812
        #how should we know if the training works if classes now start pretending to be other classes
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

    def training_step(self, batch, batch_idx):
        return self.getLosses(batch, batch_idx, train=True, test=False)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        lr = self.optimizers(use_pl_optimizer=False).param_groups[0]['lr'] 
        if batch_idx % 100 == 0:
            print("Batch:", batch_idx, "Learning rate:", lr)

    def validation_step(self, batch, batch_idx):
        return self.getLosses(batch, batch_idx,train=False, test=False)  

    def on_test_epoch_start(self):
        self.index_mapping = {}
        self.eval_vectors = []
        self.trial_label = []
        self.sprk_ids_pair = []

    def test_step(self, batch, batch_idx):
        #tiral used to compare similarity between two speakers
        trial_label, batch1, batch2 = batch
        if self.hparams.modul_name == "VisionTransformer":
            emb1, spk_id1 = self.model(batch1, train=False, test=True)
            emb2, spk_id2 = self.model(batch2, train=False, test=True)
        elif self.hparams.modul_name == "BasicTransformer":
            loss1, emb1, mel1 = self.model(batch1)
            loss2, emb2, mel2 = self.model(batch2)

        embeddings1 = emb1.detach().cpu().numpy()
        embeddings2 = emb2.detach().cpu().numpy()
        spk_id1 = spk_id1.detach().cpu().numpy()
        spk_id2 = spk_id2.detach().cpu().numpy()

        self.eval_vectors.append((embeddings1, embeddings2))
        self.sprk_ids_pair.append((spk_id1, spk_id2))
        self.index_mapping[batch_idx] = batch_idx
        self.trial_label.append(trial_label.item())

    def test_epoch_end(self, outputs):
        num_gpus = torch.cuda.device_count()

        index_mapping = {}
        if num_gpus > 1:
            eval_vectors = [None for _ in range(num_gpus)]
            dist.all_gather_object(eval_vectors, self.eval_vectors)

            table = [None for _ in range(num_gpus)]
            dist.all_gather_object(table, self.index_mapping)
            for i in table:
                index_mapping.update(i)
        else:
            eval_vectors = self.eval_vectors
            index_mapping = self.index_mapping
            sprk_ids_pair = self.sprk_ids_pair

        build_tsne(eval_vectors, sprk_ids_pair, self.speaker_encoder)
        #https://www.isca-speech.org/archive_v0/Interspeech_2017/pdfs/0803.PDF
        #mean centering
        eval_vectors = self.eval_vectors - np.mean(self.eval_vectors, axis=0)
        scores = score.cosine_score(eval_vectors)
        EER, threshold = score.compute_eer(self.trial_label, scores)

        print("\ncosine EER: {:.2f}% with threshold {:.2f}".format(EER*100, threshold))
        self.log("cosine_eer", EER*100, sync_dist=True)

        minDCF, threshold = score.compute_minDCF(self.trial_label, scores, p_target=0.01)
        print("cosine minDCF(10-2): {:.2f} with threshold {:.2f}".format(minDCF, threshold))
        self.log("cosine_minDCF(10-2)", minDCF, sync_dist=True)

    def configure_optimizers(self):
        if self.hparams.modul_name == "VisionTransformer":
            optimizer = torch.optim.AdamW(self.parameters(), self.hparams.learning_rate,weight_decay=self.hparams.weight_decay)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )        

        lr_scheduler = None
        if self.hparams.scheduler == 'stepLR':
            lr_scheduler = StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        elif self.hparams.scheduler == 'noam':
            lr_scheduler = NoamScheduler(optimizer, self.hparams.warmup_step, self.hparams.learning_rate)
            return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        elif self.hparams.scheduler == 'cosine':
            lr_scheduler = CosineWarmupScheduler(optimizer, self.hparams.warmup_step, self.hparams.learning_rate, self.hparams.max_epochs)
        elif self.hparams.scheduler == 'plateau':
            lr_scheduler = PlateauScheduler(optimizer, self.hparams.learning_rate, self.hparams.metricTracker )
            return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        if lr_scheduler is None:
            raise Exception("Scheduler not set; options are stepLR, cosine or noam; set using --scheduler")

               
        if self.hparams.resume is not None:
            maploc = 'cpu'
            if torch.cuda.device_count() > 0:
                maploc = 'cuda'
            checkpoint = torch.load(self.hparams.resume, map_location=torch.device(maploc))
            optimizer.load_state_dict(checkpoint['optimizer_states'][0])
            lr_scheduler.load_state_dict(checkpoint['lr_schedulers'][0])
            lr_scheduler.optimizer = optimizer #just hacked somewhat together

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch'}]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_lbfgs):
        # warm up learning_rate if LR_Scheduler is used
        if (self.hparams.scheduler == 'stepLR'):
            self.warmup_LR(self, optimizer)

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    @staticmethod
    def warmup_LR(self, optimizer):
        if self.trainer.global_step < self.hparams.warmup_step:
            lr_scale = min(1., float(self.trainer.global_step +
                           1) / float(self.hparams.warmup_step))
            for idx, pg in enumerate(optimizer.param_groups):
                pg['lr'] = lr_scale * self.hparams.learning_rate
        

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        (args, _) = parser.parse_known_args()
        
        parser.add_argument("--modul_name", default="VisionTransformer")
        parser.add_argument("--num_workers", default=8, type=int)
        parser.add_argument("--seed", type=int, default=3047)
        parser.add_argument("--pretrain", action='store_true')
        parser.add_argument("--second", type=float, default=2)
        parser.add_argument('--step_size', type=int, default=1)
        parser.add_argument('--gamma', type=float, default=0.5)
        parser.add_argument("--batch_size", type=int, default=50)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--warmup_step", type=float, default=5)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--scheduler", type=str, default="cosine")
        parser.add_argument("--model_size", type=str, default="small")
        parser.add_argument("--save_dir", type=str, default="results")
        parser.add_argument("--accumulate_grad_batch", type=int, default=40)
        parser.add_argument("--hop", type=int, default=154)
        parser.add_argument('--cls_pos', action='store_true')

        parser.add_argument("--checkpoint", type=str, default=None)
        parser.add_argument("--top_n_rows", type=int, default=None)
        parser.add_argument("--resume", type=str, default=None)

        files = 'files/'
        if platform == "linux" or platform == "linux2":    
            files = 'files/Linux/'
        parser.add_argument("--train_csv_path", type=str, default=files+"train.csv")
        parser.add_argument("--train2_csv_path", type=str, default=None)
        parser.add_argument("--valid_csv_path", type=str, default=files+"valid.csv")
        parser.add_argument("--test_csv_path", type=str, default=files+"test.csv")
        parser.add_argument("--rir_csv_path", type=str, default="files/rir.csv")
        parser.add_argument("--trial_path", type=str, default="files/veri_test2.txt")

        parser.add_argument("--full_utterance", action='store_false')
        parser.add_argument("--shuffle_type", type=str, default=None)
        parser.add_argument("--fixed_segment", action='store_false')
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--plot', action='store_true')
        parser.add_argument('--aug', action='store_true',  default=False)
        parser.add_argument('--speedup', type=bool, default=True)
        return parser

def cli_main():
    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = Task.add_model_specific_args(parser)
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    metricTracker = MetricTracker()
    model = Task(**args.__dict__)
    if args.resume is not None:
        model.hparams.resume = args.resume
    else:
        model.hparams.metricTracker = metricTracker

    if args.checkpoint is not None:
        if args.pretrain:
            state_dict = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
            model.load_state_dict(state_dict, strict=True)
            print("load weight from {}".format(args.checkpoint))
        else:
            checkpoint_model = torch.load(args.checkpoint, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % args.checkpoint)
            pretrained_dict = checkpoint_model['state_dict']
            state_dict = model.state_dict()

            for k in list(pretrained_dict):
                if not state_dict.__contains__(k):
                    print(f"Removing key {k} from pretrained checkpoint")
                    del pretrained_dict[k]

            msg = model.load_state_dict(pretrained_dict, strict=False)
            print(msg)

            for k in msg.missing_keys:
                if model.state_dict().__contains__(k):
                    #initialize those weithgs manually
                    trunc_normal_(model.state_dict()[k], std=2e-5)
                    print(f"initialised key {k} manually")

    

    assert args.save_dir is not None
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss', save_top_k=100, #this was a mistake we just shoulda went after val loss
           filename="{epoch}_{train_loss:.2f}", dirpath=args.save_dir)
    #checkpoint_callback = ModelCheckpoint(monitor='valid_loss', save_top_k=1,
    #       filename="{epoch}_{valid_loss:.2f}", dirpath=args.save_dir)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # init default datamodule
    print("data augmentation {}".format(args.aug))
    dm = SPKDataModul(train_csv_path=args.train_csv_path, valid_csv_path=args.valid_csv_path, 
                       test_csv_path=args.test_csv_path, train2_csv_path=args.train2_csv_path, 
                       rir_csv_path=args.rir_csv_path, speaker_encoder=model.speaker_encoder,
                     second=args.second, aug=args.aug, batch_size=args.batch_size,
                     num_workers=args.num_workers, pairs=False,
                     top_n_rows=args.top_n_rows, trial_path=args.trial_path,
                     full_utterance=args.full_utterance,
                     fixed_segment=args.fixed_segment)
    AVAIL_GPUS = torch.cuda.device_count()

    trainer = Trainer(
            max_epochs=args.max_epochs,
            strategy='ddp', #for pretrain we could use ddp_find_unused_parameters_false
            accelerator= 'gpu' if AVAIL_GPUS > 0 else 'cpu',
            devices= AVAIL_GPUS if AVAIL_GPUS > 0 else None,
            num_sanity_val_steps=0,
            sync_batchnorm= True if AVAIL_GPUS > 0 else False,
            callbacks=[checkpoint_callback, lr_monitor], #[checkpoint_callback, lr_monitor, metricTracker]
            default_root_dir=args.save_dir,
            reload_dataloaders_every_n_epochs=1,
            accumulate_grad_batches=args.accumulate_grad_batch,
            log_every_n_steps=5,
            gradient_clip_val=5
            )

    if args.plot:
        for name, param in model.named_parameters():
            print(name)
            print(param)
            print(param.size())
            print('\n')
    if args.test:
        trainer.test(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm, ckpt_path=args.resume)

if __name__ == "__main__":
    cli_main()