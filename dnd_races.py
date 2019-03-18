import re
import sys
import time
import argparse
import csv
from copy import deepcopy

from dice_roller import Dice_Roller

#from generator import RNNLayerGenerator

RACE = 0
SUBRACE = 1
SIZE = 4
SPEED = 5
LANGUAGE = 6
STR = 7
DEX = 8
CON = 9
INT = 10
WIS = 11
CHA = 12
EX_INFO = 13
SOURCE = 14
STAT_NAMES = ['str', 'dex', 'con', 'int', 'wis', 'cha']
STAT_ABBREVIATIONS = {'str':'strength', 'dex':'dexterity', 'con':'constitution', 'int':'intelligence', 'wis':'wisdom', 'cha':'charisma'}

class Race(object):

    def __init__(self, race_name = 'Aasimar', subrace_name='Protector', stats = {}, info = 'None', size = 0, speed = 30, language = 'Common', source = 'PHB'):
        self.race_name = race_name
        self.subrace_name = subrace_name
        self.stats = {}
        self.size = size
        self.speed = speed
        self.language = language
        self.race_info = info
        self.source = source
        if stats:
            self.set_stats(stats)
        else:
            self.set_stats(stats)

    def strength(self):
        self.stats['strength']

    def dexterity(self):
        self.stats['dexterity']

    def constitution(self):
        self.stats['constitution']

    def intelligence(self):
        self.stats['intelligence']

    def wisdom(self):
        self.stats['wisdom']

    def charisma(self):
        self.stats['charisma']

    def get_race_name(self):
        return ('{0}: {1}'.format(self.race_name, self.subrace_name))

    def get_language(self):
        return self.language

    def get_stats(self):
        return self.stats

    def set_stats(self, stats):
        if stats and type(stats) is list:
            self.stats['strength'] = stats[0]
            self.stats['dexterity'] = stats[1]
            self.stats['constitution'] = stats[2]
            self.stats['intelligence'] = stats[3]
            self.stats['wisdom'] = stats[4]
            self.stats['charisma'] = stats[5]

class Character(object):

    def __init__(self, race, name = 'None', gender = 'Male'):
        self.race = deepcopy(race)
        self.name = name
        self.gender = gender

    def get_language(self):
        return self.race.language

    def get_stats(self):
        return self.race.stats

    def get_race_name(self):
        return ('{0}: {1}'.format(self.race.race_name, self.race.subrace_name))

def populate_races(desired_race=None, sub_race = '(none)', return_subraces = False):
    important_info={'first_row':True, 'race':desired_race, 'subrace':sub_race, 'undesirables':['cent/mino', 'ravnica', 'psz', 'eberron']}
    filepath = 'races.csv'
    
    return parse_csv_into_dict(filepath, important_info)

def parse_csv_into_dict(filepath, important_info):
    output_dict = {}
    with open(filepath, 'r') as csv_file:
        file_lines = csv.reader(csv_file, delimiter=',')
        for row in file_lines:
            process_row_as_race(row, output_dict, important_info)
    return output_dict

def process_row_as_race(row, races, important_info):
    if important_info['first_row']:
        important_info['first_row'] = False
        return

    if 'choose one' in row[SUBRACE].lower():
        return
    if 'none' in row[SUBRACE].lower():
        row[SUBRACE] = 'None'
    
    race = add_race_to_dict_and_return(row, races)

    populate_race_info(race, row)

def add_race_to_dict_and_return(row, races):
    race_name = row[RACE]
    subrace_name = row[SUBRACE]
    dict_race_name = '{0}:{1}'.format(race_name, subrace_name)

    if dict_race_name not in races:
        races[dict_race_name] = Race(race_name, subrace_name)

    return races[dict_race_name]
    
def populate_race_info(race, row):
    race.language = row[LANGUAGE]
    race.race_info = row[EX_INFO]
    race.source = row[SOURCE]
    race.size = row[SIZE]
    race.speed = row[SPEED]

    stats = retrieve_stats(row)
    race.set_stats(stats)

def retrieve_stats(row):
    index = STR
    stats = []

    while index <=CHA:
        if not row[index] or row[index] is '_':
            stats.append(0)
        else:
            stats.append(int(row[index]))
        index += 1
    return stats

def add_rolled_choice_to_race_stats(choice, number, character):
    stat = STAT_ABBREVIATIONS[choice.lower()]
    character.get_stats()[stat] += number

""" def generate(race='', number=1, gender=''):
    mpath = './models/rnn_layer_epoch_250.pt'
    return
    dnd = RNNLayerGenerator(model_path=mpath)
    tuples = dnd.generate(number, race.lower(), gender)

    for name_tuple in tuples:
        return (name_tuple[0] + ': ' +name_tuple[2]) """

if __name__ == '__main__':
    #generate()
    races = populate_races()

    new_character = Character(races['Aasimar:Protector'])
    auto_assign = True

    rolled_stats = Dice_Roller().roll_ndx_y_times(4, 6, 7, True)

    print(str(new_character.get_race_name()))

    print('Current Stats: ' + str(new_character.get_stats()))
    print('Rolled numbers: ' + str(rolled_stats))

    stats_assigned = {stat:0 for stat in STAT_NAMES}

    auto_idx = 0
    for number in rolled_stats:
        if auto_assign:
            add_rolled_choice_to_race_stats(STAT_NAMES[auto_idx], number, new_character)
            auto_idx+=1
            continue

        while True:
            choice = input('Choose a 3 letter stat for: {0}\n'.format(number))

            if choice not in STAT_ABBREVIATIONS:
                print('That is not a stat!')
            elif not stats_assigned[choice]:
                stats_assigned[choice]=1
                break
            else:
                print('You have already assigned that stat!')

        add_rolled_choice_to_race_stats(choice, number, new_character)

    print(str(new_character.get_stats()))