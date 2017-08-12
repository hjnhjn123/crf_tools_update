# -*- coding: utf-8 -*-

from scipy import stats
import os
from json import load
from pandas.io.json import json_normalize

# from .arsenal_nlp import *
# from .arsenal_spacy import *
from .arsenal_stats import *

HEADER_FS = ['fact', 'entity_proper_name', 'entity_type']
HEADER_SN = ['factset_entity_id', 'short_name']
HEADER_SN_TYPE = ['entity_type', 'short_name']
HEADER_SCHWEB = ['Language', 'Title', 'Type']
HEADER_EXTRACTED = ['Count', 'Token', 'POS', 'NER']
HEADER_ANNOTATION = ['TOKEN', 'NER', 'POS']

LABEL_COMPANY = ['PUB', 'EXT', 'SUB', 'PVT', 'MUT', 'UMB', 'PVF', 'HOL', 'MUC', 'TRU', 'OPD', 'PEF', 'FND', 'FNS',
                 'JVT', 'VEN', 'HED', 'UIT', 'MUE', 'ABS', 'GOV', 'ESP', 'PRO', 'FAF', 'SOV', 'COR', 'IDX', 'BAS',
                 'PRT', 'SHP']
LABEL_ANS = ['category', 'nname_en']

HDF_KEY_20170425 = ['aca', 'com_single', 'com_suffix', 'location', 'name', 'ticker', 'tfdf', 'tfidf']

NON_PPL_COM_DIC = {'U-GPE': 'O', 'B-GPE': 'O', 'I-GPE': 'O', 'L-GPE': 'O', 'B-GOV': 'O', 'L-GOV': 'O',
                   'I-GOV': 'O', 'U-ACA': 'O', 'B-ACA': 'O', 'L-ACA': 'O', 'I-ACA': 'O', 'U-GOV': 'O'}

FULL_NER_DIC = {'U-COM': 'U-COM', 'O': 'O', 'B-COM': 'B-COM', 'I-COM': 'I-COM', 'L-COM': 'L-COM', 'B-PPL': 'B-PPL',
                'I-PPL': 'I-PPL', 'L-PPL': 'L-PPL', 'U-PPL': 'U-PPL', 'U-GPE': 'O', 'B-GPE': 'O', 'I-GPE': 'O',
                'L-GPE': 'O', 'B-GOV': 'O', 'L-GOV': 'O', 'I-GOV': 'O', 'U-ACA': 'O', 'B-ACA': 'O', 'L-ACA': 'O',
                'I-ACA': 'O', 'U-GOV': 'O'}

DIC_CONLL_SPACY = {'NNP': 'PROPN', 'VBZ': 'VERB', 'JJ': 'ADJ', 'NN': 'NOUN', 'TO': 'PART', 'VB': 'VERB', '.': 'PUNCT',
                   'CD': 'NUM', 'DT': 'DET', 'VBD': 'VERB', 'IN': 'ADP', 'PRP': 'PRON', 'NNS': 'PROPN', 'VBP': 'VERB',
                   'MD': 'VERB', 'VBN': 'VERB', 'POS': 'PART', 'JJR': 'ADJ', 'O': '##', 'RB': ' PART', ',': 'PUNCT',
                   'FW': 'X', 'CC': 'CONJ', 'WDT': 'ADJ', '(': 'PUNCT', ')': 'PUNCT', ':': 'PUNCT', 'PRP$': 'ADJ',
                   'RBR': 'ADV', 'VBG': 'VERB', 'EX': 'ADV', 'WP': 'NOUN', 'WRB': 'ADV', '$': 'SYM', 'RP': 'ADV',
                   'NNPS': 'PROPN', 'SYM': 'SYM', 'RBS': 'ADV', 'UH': 'INTJ', 'PDT': 'ADJ', "''": 'PUNCT',
                   'LS': 'PUNCT', 'JJS': 'ADJ', 'WP$': 'ADJ', 'NN|SYM': 'X'}

DIC_CONLL_CRF = {'U-ORG': 'U-COM', 'U-LOC': 'U-GPE', 'B-MISC': 'O', 'L-MISC': 'O', 'B-PER': 'B-PPL', 'L-PER': 'L-PPL',
                 'B-LOC': 'B-GPE', 'L-LOC': 'L-GPE', 'U-PER': 'U-PPL', 'U-MISC': 'O', 'I-MISC': 'O', 'B-ORG': 'B-COM',
                 'I-ORG': 'I-COM', 'L-ORG': 'L-COM', 'I-PER': 'I-PPL', 'I-LOC': 'I-GPE', 'O':'O'}


DIC_CONLL_CRF_ = {'U-ORG': 'U-COM', 'U-LOC': 'U-GPE', 'B-MISC': 'B-MISC', 'L-MISC': 'L-MISC', 'B-PER': 'B-PPL',
                  'L-PER': 'L-PPL','B-LOC': 'B-GPE', 'L-LOC': 'L-GPE', 'U-PER': 'U-PPL', 'U-MISC': 'U-MISC', 'I-MISC': 'I-MISC',
                  'B-ORG': 'B-COM','I-ORG': 'I-COM', 'L-ORG': 'L-COM', 'I-PER': 'I-PPL', 'I-LOC': 'I-GPE', 'O':'O'}

##############################################################################


##############################################################################
def prepare_feature_hdf(output_f, hdf_keys, mode, *files):
    """
    If you have an HDF file on your disk, please remove it first.
    :param output_f: the output file
    :param hdf_keys: list of hdf keys
    :param files: input file, one column or two columns
    :param mode: append mode by default
    """
    datas = [pd.read_csv(f, engine='c', quoting=0) for f in files]
    df2hdf(output_f, hdf_keys, mode, *datas)


##############################################################################
def extract_entity(begin_index, end_index, ner_list, sent, end_mark='person', tag='PPL'):
    """
    :param begin_index: list, index of entity beginnings
    :param end_index: list, index of entity endings
    :param ner_list: list, list of ner tags
    :param sent: list, list of tokens
    :param end_mark: str, tag in the original text
    :param tag: str, end of ner tags
    :return: list, updated ner list
    """
    all_index = zip(begin_index, end_index)

    b_tag, i_tag, l_tag, u_tag = 'B-' + tag, 'I-' + tag, 'L-' + tag, 'U-' + tag
    entity_anchor = [i for (i, t) in enumerate(sent) if (end_mark + '|||') in t]
    # may have a punctuation after '|||'

    entity_index = [i for i in all_index if i[-1] in entity_anchor]
    for index in entity_index:
        if index[-1] - index[0] == 0:
            ner_list[index[0]] = u_tag
        elif index[-1] - index[0] == 1:
            ner_list[index[0]], ner_list[index[-1]] = b_tag, l_tag
        elif index[-1] - index[0] > 1:
            ner_list[index[0]], ner_list[index[-1]] = b_tag, l_tag
            for i in range(index[0] + 1, index[-1]):
                ner_list[i] = i_tag
    return ner_list



##############################################################################


def replalce_ner(in_file, out_file):
    df = pd.read_csv(in_file, engine='c', header=None)
    df.columns = HEADER_ANNOTATION
    df['NER'] = df['NER'].map(FULL_NER_DIC)
    df.to_csv(out_file, header=None, index=False)


##############################################################################


def convert_conll2bilou(in_f, out_f):
    df = pd.read_table(in_f, header=None, delimiter=' ', skip_blank_lines=False, skiprows=2)
    df.columns = ['TOKEN', 'POS', 'POS1', 'NER']
    df = df[df.TOKEN != '-DOCSTART-']
    tt = df[HEADER_ANNOTATION]
    tt['NER'] = tt['NER'].fillna('O')
    tt['TOKEN'] = tt['TOKEN'].fillna('##END')
    tt['POS'] = tt['POS'].fillna('NIL')
    tt['POS'] = tt['POS'].map(DIC_CONLL_SPACY)
    tt_list = [list(i) for i in zip(tt.TOKEN.tolist(), tt.NER.tolist(), tt.POS.tolist())]
    for i in range(len(tt_list)):
        if tt_list[i][1].startswith('B') and tt_list[i + 1][1].startswith('O'):
            tt_list[i][1] = tt_list[i][1].replace('B-', 'U-')
        elif tt_list[i][1].startswith('I') and tt_list[i + 1][1].startswith('O'):
            tt_list[i][1] = tt_list[i][1].replace('I-', 'L-')
    result = pd.DataFrame(tt_list)
    result.columns = HEADER_ANNOTATION
    result['NER'] = result['NER'].map(DIC_CONLL_CRF_)
    result['POS'] = result['POS'].fillna('###')
    result.to_csv(out_f, index=False, header=None)


##############################################################################,head


def extract_owler(owler_dir, out_f):
    file_dir = ['/'.join((owler_dir, f)) for f in os.listdir(owler_dir)]
    files = [pd.read_json(f, lines=True) for f in file_dir]
    df = pd.concat(files)
    companies = [i['company_details']['name'] for i in df['company_info'].tolist()]
    result = pd.DataFrame(list(set([i for i in companies if len(i.split()) == 1])))
    result.to_csv(out_f, index=False, header=None)


def swap_ner_pos(in_f):
    """
    swap the column of ner and pos
    :param in_f: [token, pos, ner]
    """
    df = pd.read_csv(in_f, engine='c', header=None)
    df.columns = ['TOKEN', 'POS', 'NER']
    df[['TOKEN', 'NER', 'POS']].to_csv(in_f, index=False, header=None)