import re
import sys
import time
import argparse
import csv

from generator import RNNLayerGenerator

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

    def __init__(self, race_name):
        self.race_name = race_name
        self.subraces = []
        self.race_name = []
        self.size = 0
        self.speed = 30
        self.language = 'Common'
        self.race_info = 'None'

    def add_subrace(self, subrace, strength, dex, con, intel, wis, cha, info, size, speed, language, source):
        self.subraces.append(Subrace(subrace, strength, dex, con, intel, wis, cha, info, size, speed, language, source))

class Subrace(object):

    def __init__(self, subrace, strength, dex, con, intel, wis, cha, info, size, speed, language, source):
        self.subrace_name = subrace
        self.strength = strength
        self.dexterity = dex
        self.constitution = con
        self.intelligence = intel
        self.wisdom = wis
        self.charisma = cha
        self.race_info = info
        self.size = size
        self.speed = speed
        self.language = language
        self.source = source

def parse_csv(desired_race=None, sub_race = '(none)', return_subraces = False):
    first_row = True
    important_info={'first_row':first_row, 'race':desired_race, 'subrace':sub_race, 'undesirables':['cent/mino', 'ravnica', 'psz', 'eberron']}
    filepath = 'races.csv'
    races = {}
    
    with open(filepath, 'r') as racefile:
        file_lines = csv.reader(racefile, delimiter=',')
        for row in file_lines:
            if important_info['first_row']:
                important_info['first_row'] = False
                continue
            race_name = row[RACE]
            if race_name not in races:
                races[race_name] = Race(race_name)
            if 'choose one' in row[SUBRACE].lower():
                continue
            race = races[race_name]
            create_new_subrace(row, race)
    
    for race in races:
        for subrace in races[race].subraces:
            print (race+': '+subrace.subrace_name)

def create_new_subrace(row, race):
    subrace = row[SUBRACE]
    size = row[SIZE]
    speed = row[SPEED]
    lang = row[LANGUAGE]
    strength = row[STR]
    con = row[CON]
    dex = row[DEX]
    intel = row[INT]
    wis = row[WIS]
    cha = row[CHA]
    info = row[EX_INFO]
    source = row[SOURCE]
    
    race.add_subrace(subrace, strength, dex, con, intel, wis, cha, info, size, speed, lang, source)

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

def print_desired_info(row):
    name = generate(row[RACE])
    return (name +' - ' +row[RACE] + ': ' +row[SUBRACE]+ ' '+get_stats(row))

def get_stats(row):
    stats = ''
    statcount = 7
    while statcount <=12:
        if row[statcount] is '' or row[statcount] is '_':
            stat = '0'
        else:
            stat = row[statcount]
        stats = stats + ' ' + stat
        statcount+=1
    
    return stats

def generate(race='', number=1, gender=''):
    mpath = './models/rnn_layer_epoch_250.pt'

    dnd = RNNLayerGenerator(model_path=mpath)
    tuples = dnd.generate(number, race.lower(), gender)

    for name_tuple in tuples:
        return (name_tuple[0] + ': ' +name_tuple[2])

if __name__ == '__main__':
    parse_csv()