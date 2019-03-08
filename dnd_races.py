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

class Race(object):

    def __init__(self, race_name, stats = {}, info = 'None', size = 0, speed = 30, language = 'Common', source = 'PHB'):
        self.race_name = race_name
        self.subraces = []
        self.stats = stats
        if stats:
            self.set_stats(stats, False)
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
        self.subraces.append(Subrace(subrace, stats, info, size, speed, language, source))

    def get_stats(self, subrace = None):
        if not subrace:
            subrace = self.subraces[0]
        return subrace.get_stats()

    def set_stats(self, stats, list = True):
        if list:
            self.stats['strength'] = stats[0]
            self.stats['dexterity'] = stats[1]
            self.stats['constitution'] = stats[2]
            self.stats['intelligence'] = stats[3]
            self.stats['wisdom'] = stats[4]
            self.stats['charisma'] = stats[5]
        else:
            self.stats = stats

class Subrace(Race):

    def __init__(self, subrace_name, stats, info, size, speed, language, source):
        Race.__init__(self, subrace_name, stats, info, size, speed, language, source)        

    def get_stats(self):
        return [self.stats]

def parse_csv(desired_race=None, sub_race = '(none)', return_subraces = False):
    first_row = True
    important_info={'first_row':first_row, 'race':desired_race, 'subrace':sub_race, 'undesirables':['cent/mino', 'ravnica', 'psz', 'eberron']}
    filepath = 'races.csv'
    races = {}
    
    with open(filepath, 'r') as racefile:
        file_lines = csv.reader(racefile, delimiter=',')
        for row in file_lines:
            process_row(row, races, important_info)

    return races

def process_row(row, races, important_info):
    if important_info['first_row']:
        important_info['first_row'] = False
        return
    race_name = row[RACE]
    if race_name not in races:
        races[race_name] = Race(race_name)
    if 'choose one' in row[SUBRACE].lower():
        return
    race = races[race_name]
    create_new_subrace(row, race)
    return

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

""" def generate(race='', number=1, gender=''):
    mpath = './models/rnn_layer_epoch_250.pt'
    return
    dnd = RNNLayerGenerator(model_path=mpath)
    tuples = dnd.generate(number, race.lower(), gender)

    for name_tuple in tuples:
        return (name_tuple[0] + ': ' +name_tuple[2]) """

if __name__ == '__main__':
    #generate()
    races = parse_csv()
    rolls = Dice_Roller().roll_ndx_y_times_drop_lowest(4, 6, 7)
    stats = races['Aasimar'].get_stats()
    print(stats)
    print(rolls)