import re
import sys
import time
import argparse
import csv
from copy import deepcopy
import random

from dice_roller import Dice_Roller

#from generator import RNNLayerGenerator

#################################################################################
# Race Constants
#################################################################################
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

#################################################################################
# Class Constants
#################################################################################
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
SUBCLASS_LEVEL = 21

#################################################################################
# Background Constants
#################################################################################
BACKGROUND_NAME = 0
BACKGROUND_SKILLS = 1
BACKGROUND_LANGUAGE = 2
BACKGROUND_TOOLS = 3
BACKGROUND_SOURCE = 4

STAT_NAMES = ['str', 'dex', 'con', 'int', 'wis', 'cha']
STAT_ABBREVIATIONS = {'str':'strength', 'dex':'dexterity', 'con':'constitution', 'int':'intelligence', 'wis':'wisdom', 'cha':'charisma'}
COMMON_LANGUAGES = ['Common', 'Dwarvish', 'Elvish', 'Giant', 'Gnomish', 'Goblin', 'Halfling', 'Orc']
EXOTIC_LANGUAGES = ['Abyssal', 'Celestial', 'Draconic', 'Deep Speech', 'Infernal', 'Primordial', 'Sylvan', 'Undercommon', 'Druidic']

#################################################################################
# Custom Classes
#################################################################################

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

    def get_languages(self):
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

class Class(object):
    def __init__(self, class_name, hit_die = 'D8', best_abilities = [], saves = [], armor = [], weapons = [], skill_count = 0, tools = [], languages = ''):
        self.class_name = class_name
        self.subclasses = []
        self.is_subclass = False
        self.parent_class_name = 'None'
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
        self.subclass_level = 1
        self.starting_hit_points = self.get_hitpoints()

    def get_best_abilities(self):
        ability_list = self.best_abilities.split(',')
        abilities = []
        if ability_list:
            ability = self.best_abilities.split(',')[0].split('or')[0]
            abilities.append(ability.strip())
        if len(ability_list) > 1:
            ability = self.best_abilities.split(',')[1]
            abilities.append(ability.strip())
        return abilities

    def get_hitpoints(self):
        return int(self.hit_die[1])

    def get_hit_die(self):
        return self.hit_die

    def get_languages(self):
        return self.languages

    def add_subclasses(self, subclass):
        self.subclasses.append(subclass)

    def get_name(self):
        if self.is_subclass:
            return '{0}: {1}'.format(self.parent_class_name, self.class_name)
        return self.class_name

    def get_base_class_name(self):
        return self.parent_class_name

    def get_subclass_level(self):
        return int(self.subclass_level)

    def is_subclass_eligible(self, character_level):
        return ((self.get_subclass_level() <= character_level) and self.subclasses)

class Character(object):

    def __init__(self, race, class_object = None, level = 1, name = 'None', gender = 'Male'):
        self.race = deepcopy(race)
        self.class_object = deepcopy(class_object)
        self.name = name
        self.gender = gender
        self.level = level
        self.hit_points = 0
        self.languages = ''
        self.tools = []
        self.background = None

    def set_languages(self):
        race_languages = self.race.get_languages()
        race_languages_list = self._convert_comma_string_to_list(race_languages)

        class_languages = self.class_object.get_languages()
        class_languages_list = self._convert_comma_string_to_list(class_languages)

        character_language_list = []

        character_language_list = self._create_language_list(character_language_list, race_languages_list, class_languages_list)
        character_language_list = self._create_language_list(character_language_list, class_languages_list, race_languages_list)

        self.languages = ','.join(character_language_list)

    def get_languages(self):
        if not self.languages:
            self.set_languages()
        return self.race.language

    def get_stats(self):
        return self.race.stats

    def get_race_name(self):
        return ('{0}: {1}'.format(self.race.race_name, self.race.subrace_name))

    def get_class_name(self):
        return (self.class_object.get_name())

    def get_hitpoints(self):
        if self.hit_points is 0:
            self.hit_points = self.set_hitpoints()
        return self.hit_points

    def set_hitpoints(self):
        lvl_1_hp = self.class_object.get_hitpoints()
        leveled_hp = sum(Dice_Roller().roll_n_d_x(self.level, self.class_object.get_hit_die()))
        return lvl_1_hp + leveled_hp

    def get_name(self):
        return self.name

    def set_background(self, background):
        self.background = background

    def get_background(self):
        return self.background

    def _create_language_list(self, out_list, list_a, list_b):
        for language_a in list_a:
            if language_a not in list_b:
                out_list.append(language_a)
        return out_list

    def _convert_comma_string_to_list(self, comma_string):
        if ',' in comma_string:
            return comma_string.split(',')
        else:
            return list(comma_string)

class Background(object):

    def __init__(self, background_name, skills, language, tools, source):
        self.background_name = background_name
        self.skills = skills
        self.language = language
        self.tools = tools
        self.source = source

#################################################################################

def populate_classes():
    important_info={'first_row':True, 'undesirables':['cent/mino', 'ravnica', 'psz', 'eberron'], 'add_subclass':False}
    filepath = 'classes.csv'

    return parse_csv_into_dict(filepath, important_info, True)

def populate_races():
    important_info={'first_row':True, 'undesirables':['cent/mino', 'ravnica', 'psz', 'eberron']}
    filepath = 'races.csv'
    
    return parse_csv_into_dict(filepath, important_info)

def populate_backgrounds():
    important_info={'first_row':True}
    filepath = 'Backgrounds.csv'
    
    # Modify this to be more flexible
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

    if 'choose one' in row[SUBCLASS].lower() or 'none' in row[SUBCLASS].lower():
        row[SUBCLASS] = 'Base Class'

    # This check is necessary in case we had a class without any subclasses
    if row[SUBCLASS] == 'Base Class':
        important_info['add_subclass'] = False

    if important_info['add_subclass']:
        class_object = add_subclass_to_class(row, classes)
        populate_class_info(class_object, row, True)
    else:
        class_object = add_class_to_dict_and_return(row, classes)
        populate_class_info(class_object, row)


    # When not adding subclasses, note we are in a base class
    if not important_info['add_subclass']:
        important_info['add_subclass'] = row[SUBCLASS] == 'Base Class'
    return

def add_subclass_to_class(row, classes):
    class_name = row[CLASS_NAME]
    subclass_name = row[SUBCLASS]

    class_object = classes[class_name]
    subclass_object = Class(subclass_name)

    class_object.add_subclasses(subclass_object)
    
    return subclass_object

def add_class_to_dict_and_return(row, classes):
    class_name = row[CLASS_NAME]

    if class_name not in classes:
        classes[class_name] = Class(class_name)

    return classes[class_name]
    
def populate_class_info(class_object, row, is_subclass = False):
    class_object.hit_die = row[HITDIE]
    class_object.best_abilities = row[BEST_ABILITIES]
    class_object.saves = row[SAVES]
    class_object.armor = row[ARMOR]
    class_object.weapons = row[WEAPONS]
    class_object.skill_count = row[SKILL_COUNT]
    class_object.tools = row[TOOLS]
    class_object.language = row[LANGUAGE]
    class_object.subclass_level = row[SUBCLASS_LEVEL]

    if is_subclass:
        class_object.is_subclass = True
        class_object.parent_class_name = row[CLASS_NAME]

#################################################################################
#  Race Functions
#################################################################################

def process_row_as_race(row, races, important_info):
    if important_info['first_row']:
        important_info['first_row'] = False
        return

    if 'choose one' in row[SUBRACE].lower():
        return
    if 'none' in row[SUBRACE].lower() or 'optional' in row[SUBRACE].lower():
        row[SUBRACE] = 'None'
    if row[SOURCE].lower() in important_info['undesirables']:
        return
    
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

def create_final_stats(rolled_stats, new_character, auto_assign=False):
    auto_idx = 0
    best_idx = 0
    stats_assigned = {stat:0 for stat in STAT_NAMES}
    stats = STAT_NAMES
    best_abilities = new_character.class_object.get_best_abilities()

    for number in rolled_stats:
        if auto_assign:
            if best_idx < len(best_abilities):
                ability = best_abilities[best_idx].lower()
                add_rolled_choice_to_race_stats(ability, number, new_character)
                stats.remove(ability)
                best_idx += 1
            else:
                add_rolled_choice_to_race_stats(stats[auto_idx], number, new_character)
                auto_idx += 1
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

""" def generate_name(race='', number=1, gender=''):
    NAME = 0
    RACE = 1
    GENDER = 2

    mpath = './models/rnn_layer_epoch_250.pt'
    dnd = RNNLayerGenerator(model_path=mpath)
    tuples = dnd.generate(number, race.lower(), gender)

    for name_tuple in tuples:
        return (name_tuple[NAME] + ': ' +name_tuple[GENDER])  """

def print_base_character_info(new_character):
    print(str(new_character.get_race_name()))
    print(str(new_character.get_class_name()))
    print(str(new_character.get_languages()))

def generate_x_characters(x, races, classes):
    for _ in range(x):
        
        character_level = random.randrange(1,5)

        random_race = random.choice(list(races.values()))
        random_class = random.choice(list(classes.values()))

        if random_class.is_subclass_eligible(character_level):
            random_class = random.choice(random_class.subclasses)

        new_character = Character(random_race, random_class)

        print_base_character_info(new_character)
        print('##############################################')

if __name__ == '__main__':
    test = False
    auto_assign = True

    races = populate_races()
    classes = populate_classes()
    
    if test:
        generate_x_characters(20, races, classes)
        exit()

    random_race = random.choice(list(races.values()))
    random_class = random.choice(list(classes.values()))
    
    # name = generate_name()

    new_character = Character(random_race, random_class)
    #new_character = Character(races['Kalashtar:None'], random_class)

    print_base_character_info(new_character)

    rolled_stats = Dice_Roller().roll_ndx_y_times(4, 6, 7, True)
    rolled_stats.sort(reverse=True)

    print('Current Stats: ' + str(new_character.get_stats()))
    print('Rolled numbers: ' + str(rolled_stats))

    create_final_stats(rolled_stats, new_character, auto_assign)

    print(str(new_character.get_stats()))
