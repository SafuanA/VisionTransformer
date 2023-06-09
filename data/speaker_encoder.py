import speechbrain as sb
import pandas as pd
from utils.Helpers import readCSV

class SpeakerEncoder():
    def __init__(self, csv_train_path, csv_valid_path, csv_test_path, csv_train2_path, test:bool):

        self.encoder = sb.dataio.encoder.CategoricalEncoder()
        fileName = './files/spkr_encoded.txt'
        if not self.encoder.load_if_possible(fileName):
            df_train, df_valid, _ = readCSV(csv_train_path, csv_valid_path, None, csv_train2_path)
            data = [df_train, df_valid]
            df_merged = pd.concat(data)
            self.loadFile(fileName, df_merged)

        self.speaker_count = len(self.encoder)
        print('speaker count: ' + str(self.speaker_count))

        if test: #just add the test label during testing so the encoder has the labels and the dataset find the correct labels
            fileName = './files/spkr_encoded_test.txt'
            if not self.encoder.load_if_possible(fileName):
                df_train, df_valid, df_test = readCSV(csv_train_path, csv_valid_path, csv_test_path, csv_train2_path)
                data = [df_train, df_valid, df_test]
                df_merged = pd.concat(data)
                self.loadFile(fileName, df_merged)
            


    def get_speaker_labels_encoded(self, labels):
        return self.encoder.encode_label_torch(labels)

    def get_speaker_label_encoded(self, label):
        return self.encoder.encode_label(label)

    def get_total_speaker_count(self):
        return self.speaker_count

    def loadFile(self, fileName, df):
        for index, row in df.iterrows():
            self.encoder.ensure_label(row["spk_id"])
        self.encoder.save(fileName)

if __name__ == "__main__":
    encoder = SpeakerEncoder('train.csv', 'valid.csv', 'train.csv', None)