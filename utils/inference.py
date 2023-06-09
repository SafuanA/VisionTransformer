from utils.Helpers import load_audio
from utils import score

from tkinter import *
from tkinter import filedialog as fd
import torch

from train import Task

def set_file_path(entry, file_type='wav'):
    ext = '.wav'
    name = 'wav'
    initial_dir = '<path>'
    if file_type == 'ckpt':
        ext = '.ckpt'
        name = 'ckpt'
        initial_dir = '<path>'
    opened_file_name = fd.askopenfilename(title='Please select one wav from your set of wav.',
                           filetypes=[(name, [ext])], initialdir=initial_dir)
    if opened_file_name:
       entry.delete(0, END)
       entry.insert(0, opened_file_name)

def compare_identities(check_point, path_wav1, path_wav2, lbl):
    assert check_point is not None
    assert path_wav1 is not None
    assert path_wav2 is not None

    wav1 = load_audio(path_wav1)
    wav2 = load_audio(path_wav2)
    wavs = torch.tensor([wav1, wav2], dtype=torch.float)

    # use model after training or load weights and drop into the production system
    model = Task.load_from_checkpoint(check_point)
    model.eval()
    threshold =  0.13

    with torch.no_grad():
        emb = model(wavs)
        res = score.cosine_score_inference(emb)
        lbl.set(str(res) + '- same speaker: '+ str(res > threshold))


def buildGUI():
    START = "1.0"
    END = "end" #constant needed for openeing/deleting till end
    gui = Tk()
    gui.geometry("1100x350")

    txtChk_var=StringVar()
    txtChk_var.set('<path>')
    txtSrc_var=StringVar()
    txtSrc_var.set('<path>')
    txtTgt_var=StringVar()
    txtTgt_var.set('<path>')
    txtScr_var=StringVar()


    lblChk = Label(text="Checkpoint")
    lblChk.grid(row=0, column=0)
    txtChk = Entry(text="...", textvariable=txtChk_var, width=90)
    txtChk.grid(row=0, column=1)
    btnChk = Button(text="Checkpoint Laden", command=lambda:set_file_path(txtChk, 'ckpt'))
    btnChk.grid(row=0, column=2)

    lblSrc = Label(text="Quelle 1 (wav)")
    lblSrc.grid(row=1, column=0)
    txtSrc = Entry(text="...", textvariable=txtSrc_var, width=90)
    txtSrc.grid(row=1, column=1)
    btnSrc = Button(text="File 1 Laden", command=lambda:set_file_path(txtSrc))
    btnSrc.grid(row=1, column=2)

    lblTgt = Label(text="Quelle 2 (wav)")
    lblTgt.grid(row=2, column=0)
    txtTgt = Entry(text="...", textvariable=txtTgt_var, width=90)
    txtTgt.grid(row=2, column=1)
    btnTgt = Button(text="File 2 Laden", command=lambda:set_file_path(txtTgt))
    btnTgt.grid(row=2, column=2)


    btnResult = Button(text="Vergleichen", command=lambda:compare_identities(txtChk_var.get(), txtSrc_var.get(), txtTgt_var.get(), txtScr_var))
    btnResult = btnResult.grid(row=3, column=1)

    lblSore = Label(text="Score:")
    lblSore.grid(row=5, column=0)
    txtScr = Entry(text="", textvariable=txtScr_var, width=100)
    txtScr.grid(row=5, column=1)

    gui.mainloop()

if __name__ == "__main__":
    buildGUI()
