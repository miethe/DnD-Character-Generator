import re
import sys
import time
import argparse
import csv

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

    def __init__(self, race_name = 'Aasimar', stats = {}, info = 'None', size = 0, speed = 30, language = 'Common', source = 'PHB'):
        self.race_name = race_name
        self.subraces = []
        self.stats = stats
        if stats:
            self.set_stats(stats)
        else:
            self.set_stats(stats)
        self.size = size
        self.speed = speed
        self.language = language
        self.race_info = info
        self.source = source

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

    def add_subrace(self, subrace, stats, info, size, speed, language, source):
        self.subraces.append(Subrace(self.race_name, subrace, stats, info, size, speed, language, source))

    def get_stats(self, subrace = None):
        if not subrace:
            subrace = self.subraces[0]
        return subrace.get_stats()

    def set_stats(self, stats):
        if stats and stats is list:
            self.stats['strength'] = stats[0]
            self.stats['dexterity'] = stats[1]
            self.stats['constitution'] = stats[2]
            self.stats['intelligence'] = stats[3]
            self.stats['wisdom'] = stats[4]
            self.stats['charisma'] = stats[5]
        else:
            self.stats = stats

class Subrace(Race):

    def __init__(self, race, subrace = 'None', stats = {}, info = 'None', size = 0, speed = 30, language = 'Common', source = 'PHB'):
        Race.__init__(self, race, stats, info, size, speed, language, source)
        self.subrace_name = subrace
        self.race = race

    def get_stats(self):
        return self.stats

class Character(object):

    def __init__(self, subrace = Subrace('Aasimar', 'None'), name = 'None', gender = 'Male'):
        self.subrace = subrace
        self.name = name
        self.stats = subrace.get_stats()
        self.gender = gender
        self.language = subrace.language

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

    def set_stats(self, stats, list = True):
        if list and stats:
            self.stats['strength'] = stats[0]
            self.stats['dexterity'] = stats[1]
            self.stats['constitution'] = stats[2]
            self.stats['intelligence'] = stats[3]
            self.stats['wisdom'] = stats[4]
            self.stats['charisma'] = stats[5]
        else:
            self.stats = stats

    def get_stats(self):
        return self.stats

def parse_csv_into_dict(filepath, important_info):
    output_dict = {}
    with open(filepath, 'r') as csv_file:
        file_lines = csv.reader(csv_file, delimiter=',')
        for row in file_lines:
            process_row_as_race(row, output_dict, important_info)
    return output_dict

def populate_races(desired_race=None, sub_race = '(none)', return_subraces = False):
    important_info={'first_row':True, 'race':desired_race, 'subrace':sub_race, 'undesirables':['cent/mino', 'ravnica', 'psz', 'eberron']}
    filepath = 'races.csv'
    
    return parse_csv_into_dict(filepath, important_info)

def process_row_as_race(row, races, important_info):
    if important_info['first_row']:
        important_info['first_row'] = False
        return
    race_name = row[RACE]
    if race_name not in races:
        races[race_name] = Race(race_name)
    
    race = races[race_name]
    populate_race_info(race, row)

    if 'choose one' in row[SUBRACE].lower():
        return
    else:
        create_new_subrace(row, race)
    return

def populate_race_info(race, row):
    race.language = row[LANGUAGE]
    race.race_info = row[EX_INFO]
    race.source = row[SOURCE]
    race.size = row[SIZE]
    race.speed = row[SPEED]

def get_stats(row):
    index = STR
    stats = []

    while index <=CHA:
        if not row[index] or row[index] is '_':
            stats.append(0)
        else:
            stats.append(int(row[index]))
        index += 1
    return stats

def create_new_subrace(row, race):
    subrace = row[SUBRACE]
    size = row[SIZE]
    speed = row[SPEED]
    lang = row[LANGUAGE]
    info = row[EX_INFO]
    source = row[SOURCE]

    stats = get_stats(row)
    
    race.add_subrace(subrace, stats, info, size, speed, lang, source)

def skip_if_filtered(row, important_info):
    if important_info['first_row']:
        important_info['first_row'] = False
        return True
    if len(row) <=1:
        return True
    if row[SOURCE].lower().split(' ')[0] in important_info['undesirables']:
        return True
    if 'choose one' in row[SUBRACE].lower():
        return True

    if important_info['race'] is None:
        return False
    elif important_info['race'].lower() not in row[RACE].lower():
        return True

    if important_info['subrace'] is 'all':
        return False
    elif important_info['subrace'].lower() not in row[SUBRACE].lower():
        return True
    return False

def add_rolled_choice_to_race_stats(choice, number, character):
    stat = STAT_ABBREVIATIONS[choice.lower()]
    character.stats[stat] += number

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
    new_character = Character(races['Aasimar'])
    auto_assign = True

    rolled_stats = Dice_Roller().roll_ndx_y_times(4, 6, 7, True)

    #race_stats = new_character.get_stats()
    #new_character.subrace.set_stats(race_stats, True)

    print(str(new_character.subrace.race_name))
    print(str(new_character.language))

    print('Current Stats: ' + str(new_character.stats))
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

    print(str(new_character.stats))