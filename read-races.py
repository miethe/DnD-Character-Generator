import csv
from flask import request, render_template, flash, redirect, Flask
import re
import sys
import time
import argparse

from generator import RNNLayerGenerator
from train import TrainerFactory

from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

class ReusableForm(Form):
    race = TextField('Race:', validators=[validators.required()])
    subrace = TextField('Subrace:', validators=[validators.required()])

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

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

subraces = ['None']

@app.route('/', methods=['GET', 'POST'])
def default_response():
    return redirect("/create_character")
    
@app.route('/create_character', methods=['GET', 'POST'])
def create_character_from_flask():
    return provide_create_character_webform()

def process_request(request, subrace = None):
    global subraces
    get_subraces = False
    race = request.form['race']
    if subrace is None:
        subrace = request.form['subrace']
    else:
        get_subraces = True
    output = parse_csv(race, subrace, get_subraces)
    if get_subraces:
        subraces = output
    return output

def provide_create_character_webform():
    form = ReusableForm(request.form)
    if request.method == 'POST':
        if request.form['submit_button'] == 'createchar' and form.validate():
            flash(process_request(request))
        elif request.form['submit_button'] == 'getsubraces':
            flash(process_request(request, 'all'))
    else:
        flash('All the form fields are required. ')
    return render_template('race_selector.html', form=form, option_list=subraces)

###########################################################################################################

def parse_csv(desired_race=None, sub_race = '(none)', return_subraces = False):
    first_row = True
    important_info={'first_row':first_row, 'race':desired_race, 'subrace':sub_race, 'undesirables':['cent/mino', 'ravnica', 'psz', 'eberron']}
    filepath = 'races.csv'
    to_print = []
    subraces = []
    
    with open(filepath, 'r') as racefile:
        file_lines = csv.reader(racefile, delimiter=',')
        for row in file_lines:
            if skip_if_filtered(row, important_info):
                continue
            to_print.append(print_desired_info(row))
            if return_subraces:
                subraces.append(row[SUBRACE])
        if return_subraces:
            return subraces
        if len(to_print) is 1:
            return to_print[0]
        return to_print

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
    app.run()
    #parse_csv()