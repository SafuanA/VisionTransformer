from  utils.VisionTransformerPlotting import CreateVisionTransformerSmall, CreateVisionTransformer
import torch
from data.speaker_encoder import SpeakerEncoder
from utils.plotting import plot_features
from timm.models.layers import trunc_normal_
from utils.Helpers import readCSV
from data.dataset import TrainDataset
from matplotlib import pyplot as plt
import numpy as np

class analysis(torch.nn.Module):
    def __init__(self,  model):
        super().__init__()
        self.model = model

    def loadCheckPoint(self, path):
        checkpoint_model = torch.load(path, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % path)
        pretrained_dict = checkpoint_model['state_dict']
        state_dict = self.state_dict()

        for k in list(pretrained_dict):
            if not state_dict.__contains__(k):
                print(f"Removing key {k} from pretrained checkpoint")
                del pretrained_dict[k]

        msg = self.load_state_dict(pretrained_dict, strict=True)
        print(msg)

        for k in msg.missing_keys:
            if model.state_dict().__contains__(k):
                #initialize those weithgs manually
                trunc_normal_(model.state_dict()[k], std=2e-5)
                print(f"initialised key {k} manually")

    def loadDataModul(self):
        path = '<path>'
        train, valid, test = path+'train.csv', path+'valid.csv', path+'test.csv'
        df_train, df_valid, df_test = readCSV(train, valid, test, None)
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.speaker_encoder = SpeakerEncoder(train,valid,test, None, False)
        self.train_dataset = TrainDataset(df_train, self.speaker_encoder, 2, False,
                                    False, 100, None, None, 
                                    full_utterance=True, fixed_audio_start=True)

    def plot_attention_maps(self, model):
        counter = 0
        self.plot_embed(model.embedcached)
       
        for block in model.blocks:
            counter += 1;
            if counter == 1 or counter == 12:
                attn = block.attn.attnLgtCache
                self.plot_logits(attn, counter)
                attn = block.attn.attnCache
                self.plot_attn(attn, counter)
                attn = block.attn.attnCache2
                self.plot_attn_cat(attn, counter, 'attn_cat')

    def plot_embed(self, embed, name='patch embeddings'):
        embed = embed.cpu().numpy() 
        embed = embed[0]
        x_len = len(embed[0,:])
        y_len = len(embed[:,0])
        fig, ax = plt.subplots()
        ax.imshow(embed, origin='upper') #, vmin=0, cmap='jet')
        plt.savefig('plots/'+name)
        plt.clf()


    def plot_attn_cat(self, attn_maps, block:int, name='attn_cat'):
        attn_maps = attn_maps.cpu().numpy() 
        num_heads=1
        seq_len = attn_maps.shape[-2]
        seq_len2 = attn_maps.shape[-1]
        fig_size = 3
        fig, ax = plt.subplots(1, num_heads, figsize=(num_heads*fig_size, fig_size))

        label = list(range(seq_len))
        label2 = list(range(seq_len2))

        ax.imshow(attn_maps[0], origin='upper') #, vmin=0)
        ax.set_xticks(label2[::40])
        ax.set_yticks(label[::40])
        ax.set_title(f"Block:{block}")

        plt.savefig('plots/'+name+str(block))
        plt.clf()

    def plot_attn(self, attn_maps, block:int, name='attn'):
        attn_maps = attn_maps.cpu().numpy() 

        num_heads=3
        seq_len = attn_maps.shape[-2]
        seq_len2 = attn_maps.shape[-1]
        fig_size = 3
        fig, ax = plt.subplots(1, num_heads, figsize=(num_heads*fig_size, fig_size))

        label = list(range(seq_len))
        label2 = list(range(seq_len2))
        for column in range(num_heads):
            ax[column].imshow(attn_maps[0][column], origin='upper') #, vmin=0)
            ax[column].set_xticks(label2[::40])
            ax[column].set_yticks(label[::40])
            ax[column].set_title(f"Block:{block}, Head {column+1}")
        plt.savefig('plots/'+name+str(block))
        plt.clf()

    def plot_logits(self, attn_maps, block:int, name='attn_logits'):
        attn_maps = attn_maps.cpu().numpy() 

        num_heads = attn_maps.shape[1]
        seq_len = attn_maps.shape[2]
        fig_size = 4 if num_heads == 1 else 3
        fig, ax = plt.subplots(1, num_heads, figsize=(num_heads*fig_size, fig_size))

        label = list(range(seq_len))
        for column in range(num_heads):
            ax[column].imshow(attn_maps[0][column], origin='upper') #, vmin=0)
            ax[column].set_xticks(label[::40])
            ax[column].set_yticks(label[::40])
            ax[column].set_title(f"Block:{block}, Head {column+1}")
        plt.savefig('plots/'+name+str(block))
        plt.clf()

    def plotLrLoss(self):
        # example values
        data = np.genfromtxt('plots/val_history.csv', delimiter=',', skip_header=1)
        #data[:,2] = np.log2(data[:,2])/np.log(1.2)
        #data[:,3] = np.log2(data[:,3])/np.log(1.2)
        label = ['OS_l','OSP_l','OS_acc','OSP_acc']

        plt.plot(data[:,0:2], '-o', markersize=2)
        plt.legend(label[0:2])
        plt.savefig('plots/history_loss')
        plt.clf()

        plt.plot(data[:,2:4], '-o', markersize=2)
        plt.legend(label[2:4])
        plt.savefig('plots/history_acc')
        plt.clf()

    def plot_positional_encoding(self, pe, full=None):
        #sns.set_theme()
        pe = pe.squeeze().T
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,3))
        pos = ax.imshow(pe, cmap="RdGy", extent=(1,pe.shape[1]+1,pe.shape[0]+1,1))
        fig.colorbar(pos, ax=ax)
        ax.set_xlabel("Position in sequence")
        ax.set_ylabel("Hidden dimension")
        ax.set_title("Positional encoding over hidden dimensions")
        ax.set_xticks([1]+[i*10 for i in range(1,1+pe.shape[1]//10)])
        ax.set_yticks([1]+[i*10 for i in range(1,1+pe.shape[0]//10)])
        plt.show()

        if full:
            fig, ax = plt.subplots(2, 2, figsize=(12,4))
            ax = [a for a_list in ax for a in a_list]
            for i in range(len(ax)):
                ax[i].plot(np.arange(1,17), pe[i,:16], color=f'C{i}', marker="o", markersize=6, markeredgecolor="black")
                ax[i].set_title(f"Encoding in hidden dimension {i+1}")
                ax[i].set_xlabel("Position in sequence", fontsize=10)
                ax[i].set_ylabel("Positional encoding", fontsize=10)
                ax[i].set_xticks(np.arange(1,17))
                ax[i].tick_params(axis='both', which='major', labelsize=10)
                ax[i].tick_params(axis='both', which='minor', labelsize=8)
                ax[i].set_ylim(-1.2, 1.2)
            fig.subplots_adjust(hspace=0.8)
            #sns.reset_orig()
            plt.show()

if __name__ == "__main__":
    print('Begin Analysis')
    seed = 3047
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CreateVisionTransformerSmall(False, 1211, 'OS', 2)
    analysis = analysis(model)
    analysis.loadDataModul()

    #path = 'results/Try1/epoch=79_train_loss=0.45.ckpt'
    path = 'results/Try5/finetune/epoch=79_train_loss=0.83.ckpt'
    analysis.loadCheckPoint(path)

    wav, label, spk_id_encoded, path  = analysis.train_dataset.getRowDebug(0)
    wav = torch.Tensor(wav)
    wav = wav.unsqueeze(0)
    spk_id_encoded = torch.LongTensor([spk_id_encoded], device=torch.device('cpu'))

    analysis.plotLrLoss()

    model.eval()
    with torch.no_grad():
        x = model((wav, label, spk_id_encoded, path))
    analysis.plot_attention_maps(model)
    plot_features(model.melcached[0], name='analysis', transpose=True)


