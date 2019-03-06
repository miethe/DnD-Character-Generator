import numpy as np
import torch
from torch.distributions import OneHotCategorical
from torchvision.transforms import Compose

from data import Vocabulary, OneHot, Genders, Races, ToTensor
from utils import load_model


class Generator:
    """Base Generator class that can load trained model and require every subclass to implement `generate` method"""
    def __init__(self, model_path, device="cpu"):
        self.model = load_model(model_path, device=device)
        self.device = device

    def generate(self, num_samples):
        raise NotImplementedError


class RNNCellGenerator(Generator):
    def __init__(self, model_path, device="cpu"):
        super().__init__(model_path, device)

        self.vocab = Vocabulary()
        self.races = Races()
        self.genders = Genders()
        self.to_tensor = ToTensor()

        self.name_transform = Compose([self.vocab, OneHot(self.vocab.size), ToTensor()])
        self.race_transform = Compose([self.races, OneHot(self.races.size), ToTensor()])
        self.gender_transform = Compose([self.genders, OneHot(self.genders.size), ToTensor()])

    def _init_random_input(self):
        """Helper function that initialize random letter, race and gender"""
        letter = np.random.choice(self.vocab.start_letters)
        race = np.random.choice(self.races.available_races)
        gender = np.random.choice(self.genders.available_genders)

        return letter, race, gender

    def _transform_input(self, letter, race, gender):
        """Helper function to transform input into tensors"""
        letter_tensor = self.name_transform(letter).to(self.device)
        race_tensor = self.race_transform(race).to(self.device)
        gender_tensor = self.gender_transform(gender).to(self.device)

        return letter_tensor, race_tensor, gender_tensor

    def generate(self, num_samples):
        with torch.no_grad():
            print("_" * 20)
            for _ in range(num_samples):
                hx, cx = self.model.init_states(batch_size=1, device=self.device)

                letter, race, gender = self._init_random_input()
                letter_t, race_t, gender_t = self._transform_input(letter, race, gender)

                input = torch.cat([letter_t, race_t, gender_t], 1)
                outputs = [letter]

                while True:
                    output, hx, cx = self.model(input, hx, cx)

                    sample = OneHotCategorical(logits=output).sample()
                    index = torch.argmax(sample)
                    char = self.vocab.idx2char[index.item()]
                    outputs.append(char)

                    input = torch.cat([sample, race_t, gender_t], 1)

                    if char == '.' or len(outputs) == 50:
                        break

                print("Start letter: {}, Race: {}, Gender: {}".format(letter, race, gender))
                print("Generated sample: {}".format(''.join(map(str, outputs))))

            print("_" * 20)


class RNNLayerGenerator(Generator):
    def __init__(self, model_path, device="cpu", max_len=50, verbose=1):
        super().__init__(model_path, device)
        self.max_len = max_len
        self.verbose = verbose

        self.vocab = Vocabulary()
        self.races = Races()
        self.genders = Genders()
        self.to_tensor = ToTensor()

        self.name_transform = Compose([self.vocab, OneHot(self.vocab.size), ToTensor()])
        self.race_transform = Compose([self.races, OneHot(self.races.size), ToTensor()])
        self.gender_transform = Compose([self.genders, OneHot(self.genders.size), ToTensor()])

    def _init_random_input(self, skip_ran_gen=[]):
        """Helper function that initialize random letter, race and gender"""
        ran_opt = ['letter', 'race', 'gender']
        letter = ''
        gender = ''
        race = ''
        
        if not skip_ran_gen:
            letter = np.random.choice(self.vocab.start_letters)
            race = np.random.choice(self.races.available_races)
            gender = np.random.choice(self.genders.available_genders)
        else:
            for i in ran_opt:
                if i not in skip_ran_gen:
                    if i is 'letter':
                        letter = np.random.choice(self.vocab.start_letters)
                    elif i is 'race':
                        race = np.random.choice(self.races.available_races)
                    elif i is 'gender':
                        gender = np.random.choice(self.genders.available_genders)
        return letter, race, gender

    def _transform_input(self, letter, race, gender):
        """Helper function to transform input into tensors"""
        letter_tensor = self.name_transform(letter).to(self.device)
        race_tensor = self.race_transform(race).to(self.device)
        gender_tensor = self.gender_transform(gender).to(self.device)

        return letter_tensor, race_tensor, gender_tensor

    def _expand_dims(self, *tensors):
        """Add dimension along 0-axis to tensors"""
        return [torch.unsqueeze(t, 0) for t in tensors]

    def sample(self, letter, race, gender):
        """Sample name from start letter, race and gender"""
        with torch.no_grad():
            assert letter in self.vocab.start_letters, "Invalid letter"
            assert race in self.races.available_races, "Invalid race"
            assert gender in self.genders.available_genders, "Invalid gender"

            # Prepare inputs
            letter_t, race_t, gender_t = self._transform_input(letter, race, gender)
            letter_t, race_t, gender_t = self._expand_dims(letter_t, race_t, gender_t)

            # Merge all input tensors
            input = torch.cat([letter_t, race_t, gender_t], 2)
            outputs = [letter]

            # Initialize hidden states
            hx, cx = self.model.init_states(batch_size=1, device=self.device)

            while True:
                output, hx, cx = self.model(input, hx, cx, lengths=torch.tensor([1]))

                sample = OneHotCategorical(logits=output).sample()
                index = torch.argmax(sample)
                char = self.vocab.get_char(index.item())

                if char == '.' or len(outputs) == self.max_len:
                    break

                outputs.append(char)
                input = torch.cat([sample, race_t, gender_t], 2)

            name = ''.join(map(str, outputs))
            return name

    def generate(self, num_samples, in_race, in_gender):
        """Sample random names"""
        gen_names = []
        ran_gen_names = []
        if in_race is not '':
            ran_gen_names.append('race')
        if in_gender is not '':
            ran_gen_names.append('gender')
        
        for _ in range(num_samples):
            letter, race, gender = self._init_random_input(ran_gen_names)
            race = race + in_race
            gender = gender + in_gender
            gen_name = self.sample(letter, race, gender)
            gen_names.append([gen_name, race, gender])

        return gen_names


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--model_path")
    parser.add_argument("-race")
    parser.add_argument("-number")
    parser.add_argument("-gender")
    args = parser.parse_args()

    if args.number:
        number = int(args.number)
    else:
        number = 5
    if args.race:
        race = args.race
    else:
        race = ''
    if args.gender:
        gender = args.gender
    else:
        gender = ''

    dnd = RNNLayerGenerator(model_path="./models/rnn_layer_epoch_250.pt")
    tuples = dnd.generate(number, race, gender)

    for name_tuple in tuples:
        print (name_tuple[0])
