import re
import sys
import time
import argparse
import csv
from copy import deepcopy
import random

from dice_roller import Dice_Roller

#from generator import RNNLayerGenerator

# Race Constants
RACE = 0
SUBRACE = 1
SIZE = 3
SPEED = 4
LANGUAGE = 5
STR = 6
DEX = 7
CON = 8
INT = 9
WIS = 10
CHA = 11
EX_INFO = 12
SOURCE = 13

# Class Constants
CLASS_NAME = 0
SUBCLASS = 1
HITDIE = 2
BEST_ABILITIES = 7
SAVES = 8
ARMOR = 9
WEAPONS = 10
SKILL_COUNT = 11
TOOLS = 12
CLASS_LANGUAGES = 13

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

    def __init__(self, race, c_class = None, name = 'None', gender = 'Male', level = 1):
        self.race = deepcopy(race)
        self.c_class = deepcopy(c_class)
        self.name = name
        self.gender = gender
        self.level = level
        self.hit_points = 0

    def get_language(self):
        return self.race.language

    def get_stats(self):
        return self.race.stats

    def get_race_name(self):
        return ('{0}: {1}'.format(self.race.race_name, self.race.subrace_name))

    def get_hitpoints(self):
        if self.hit_points is 0:
            self.hit_points = self.set_hitpoints()
        return self.hit_points

    def set_hitpoints(self):
        lvl_1_hp = self.c_class.get_hitpoints()
        leveled_hp = sum(Dice_Roller().roll_n_d_x(self.level, self.c_class.get_hit_die()))
        return lvl_1_hp + leveled_hp

class Class(object):
    def __init__(self, class_name, subclass_name, hit_die = 'D8', best_abilities = [], saves = [], armor = [], weapons = [], skill_count = 0, tools = [], languages = ''):
        self.class_name = class_name
        self.subclass_name = subclass_name
        self.hit_die = hit_die

        # can use for improved auto character gens
        self.best_abilities = best_abilities

        self.saves = saves
        self.armor = armor
        self.weapons = weapons
        # will need to add skill list
        self.skill_count = skill_count
        self.tools = tools
        self.languages = languages
        self.starting_hit_points = self.get_hitpoints()

    def get_hitpoints(self):
        return int(self.hit_die[1])

    def get_hit_die(self):
        return self.hit_die

def populate_classes():
    important_info={'first_row':True, 'undesirables':['cent/mino', 'ravnica', 'psz', 'eberron']}
    filepath = 'classes.csv'

    return parse_csv_into_dict(filepath, important_info, True)

def populate_races():
    important_info={'first_row':True, 'undesirables':['cent/mino', 'ravnica', 'psz', 'eberron']}
    filepath = 'races.csv'
    
    return parse_csv_into_dict(filepath, important_info)

def parse_csv_into_dict(filepath, important_info, classes = False):
    output_dict = {}
    with open(filepath, 'r') as csv_file:
        file_lines = csv.reader(csv_file, delimiter=',')
        for row in file_lines:
            if classes:
                process_row_as_class(row, output_dict, important_info)
            else:
                process_row_as_race(row, output_dict, important_info)
    return output_dict

#################################################################################
# Character Class Functions
#################################################################################

def process_row_as_class(row, classes, important_info):
    if important_info['first_row']:
        important_info['first_row'] = False
        return

    c_class = add_class_to_dict_and_return(row, classes)

    populate_class_info(c_class, row)

def add_class_to_dict_and_return(row, classes):
    class_name = row[CLASS_NAME]
    subclass_name = row[SUBCLASS]
    if not row[SUBCLASS]:
        subclass_name = 'None'
    dict_class_name = '{0}:{1}'.format(class_name, subclass_name)

    if dict_class_name not in races:
        races[dict_class_name] = Class(class_name, subclass_name)

    return classes[dict_class_name]
    
def populate_class_info(c_class, row):
    c_class.hit_die = row[HITDIE]
    c_class.best_abilities = row[BEST_ABILITIES]
    c_class.saves = row[SAVES]
    c_class.armor = row[ARMOR]
    c_class.weapons = row[WEAPONS]
    c_class.skill_count = row[SKILL_COUNT]
    c_class.tools = row[TOOLS]
    c_class.language = row[LANGUAGE]

#################################################################################
#  Race Functions
#################################################################################

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

    new_character = Character(random.choice(list(races.values())))
    #new_character = Character(races['Aasimar:Protector'])
    auto_assign = True

    rolled_stats = Dice_Roller().roll_ndx_y_times(4, 6, 7, True)

    print(str(new_character.get_race_name()))
    print(str(new_character.get_language()))

    print('Current Stats: ' + str(new_character.get_stats()))
    print('Rolled numbers: ' + str(rolled_stats))

    stats_assigned = {stat:0 for stat in STAT_NAMES}

    auto_idx = 0
    for number in rolled_stats:
        if auto_assign:
            # Can use best_abilities in classes here to set ideal stats
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