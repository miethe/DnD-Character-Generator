from generator import RNNLayerGenerator
from train import TrainerFactory

def train():

    trainer = TrainerFactory.get_trainer(trainer_type="layer",
                root_dir='./data',
                epochs=300,
                batch_size=128,
                lr=0.0001,
                device="cpu",
                logfile="train_loss.log",
                verbose=1)
    trainer.run_train_loop()

def generate():
    number = 5
    race = ''
    gender = ''
    mpath = './models/rnn_layer_epoch_250.pt'

    dnd = RNNLayerGenerator(model_path=mpath)
    tuples = dnd.generate(number, race, gender)

    for name_tuple in tuples:
        print (name_tuple[0] + ': ' +name_tuple[2] + ' ' + name_tuple[1])

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-train")
    args = parser.parse_args()

    if args.train:
        train()

    generate()