# -*- coding: utf-8 -*-

import joblib as jl
import pandas as pd
import datetime

from .arsenal_crf import process_annotated, batch_add_features, batch_loading, feed_crf_trainer, df2crfsuite, \
     module_crf_train, module_crf_fit, tag_convert, token_text, token_generate
from .arsenal_logging import basic_logging
from .arsenal_test import evaluate_ner_result
from .settings import *

from .arsenal_crf import crf_predict
# from .arsenal_stats import get_now


##############################################################################


# Training


def pipeline_train(train_f, test_f, model_f, result_f, hdf_f, hdf_key, feature_conf, window_size, col_names):
    """
    A pipeline for CRF training
    :param train_f: train dataset in a 3-column csv (TOKEN, LABEL, POS)
    :param test_f: test dataset in a 3-column csv (TOKEN, LABEL, POS)
    :param model_f: model file
    :param feature_conf: feature configurations
    :param hdf_f: feature HDF5 file
    :param hdf_key: keys of feature HDF5 file
    """
    basic_logging('loading conf begins')
    f_dics = batch_loading(hdf_f, hdf_key)
    basic_logging('loading conf ends')
    train_df = tag_convert(train_f)
    test_df = tag_convert(test_f)
    # train_df, test_df = process_annotated(train_f, col_names), process_annotated(test_f, col_names)
    basic_logging('loading data ends')
    crf, _, _ = module_crf_train(train_df, f_dics, feature_conf, hdf_key, window_size)
    # test_df = pd.read_table(test_f)
    y_pred, _, _, index_line = module_crf_fit(test_df, crf, f_dics, feature_conf, hdf_key, window_size, result_f,line=False)
    y_pred = [i for j in y_pred for i in j]
    token_text(test_df, y_pred, index_line)
    if model_f:
        jl.dump(crf, model_f)
    return crf


def pipline_predict(test_f, model_f, hdf_f, hdf_key, feature_conf, window_size):
    '''
    tokenizer the text line by line
    '''
    basic_logging('loading conf begins')
    model = jl.load(model_f)
    f_dics = batch_loading(hdf_f, hdf_key)
    basic_logging('loading conf ends')

    test_line_df = pd.read_table(test_f, header=None)
    test_line_df.columns = ['TOKEN']
    text_list = []
    for line in test_line_df["TOKEN"].tolist():
        line_list = list(line.strip('\n').strip())
        line_df = pd.DataFrame(line_list)
        line_df.columns = ["TOKEN"]
        y_pred, _, _,_ = module_crf_fit(line_df, model, f_dics, feature_conf, hdf_key, window_size, '',line=True)
        y_pred = [j for i in y_pred for j in i]
        line_tag_df = tuple(list(zip(line_list, y_pred)))
        phrases = token_generate(line_tag_df)
        text_list.append(' '.join(phrases))
    pd.DataFrame(text_list).to_csv(PREDICT_FILE, header=False, index=False)
    basic_logging('converting results ends')
    return text_list



##############################################################################



##############################################################################

# Validation


def pipeline_validate(valid_f, model_f, feature_conf, hdf_f, result_f, hdf_key, window_size, col_names):
    """
    A pipeline for CRF validating
    :param valid_df: validate dataset with at least two columns (TOKEN, LABEL)
    :param model_f: model file
    :param feature_conf: feature configurations
    :param hdf_f: feature HDF5 file
    :param hdf_key: keys of feature HDF5 file
    :param window_size:
    """
    valid_df = process_annotated(valid_f, col_names)
    basic_logging('loading conf begins')
    crf = jl.load(model_f)
    f_dics = batch_loading(hdf_f, hdf_key)
    basic_logging('loading conf ends')
    y_pred, X_test, y_test = module_crf_fit(valid_df, crf, f_dics, feature_conf, hdf_key, window_size, result_f)
    result, _ = evaluate_ner_result(y_pred, y_test)
    result.to_csv(result_f, index=False)
    return result


##############################################################################


def module_batch_annotate_single_model(prepared_df, model_f, hdf_f, hdf_key, feature_conf, window_size):
    """
    :param prepared_df: a df with at-least two columns
    :param out_f: CSV FILE, the ouptut file
    :param model_f: NUMPY PICKLE FILE, the model file
    :param hdf_f: HDF FILE, the hdf file of feature dicts or lists
    :param hdf_key: LIST, the key to extract hdf file
    :param feature_conf: DICT, features used to compute
    :param window_size: INT, set window size for CRF tagging
    :param col_names: LIST, the column in json file to be used
    """
    basic_logging('loading conf begins')
    model = jl.load(model_f)
    f_dics = batch_loading(hdf_f, hdf_key)
    basic_logging('loading conf ends')

    # raw_list = (pd.read_json('/'.join((in_folder, in_f)), col_names) for in_f in listdir(in_folder))
    # basic_logging('reading files ends')

    y_pred, _, _ = module_crf_fit(prepared_df, model, f_dics, feature_conf, hdf_key, window_size, '')

    recovered_pred = [i + ['O'] for i in y_pred]
    crf_result = [i for j in recovered_pred for i in j]
    final_result = pd.concat([prepared_df[0], pd.DataFrame(crf_result), prepared_df[2]], axis=1)
    basic_logging('converting results ends')
    return pd.DataFrame(final_result)

##############################################################################


RESULT_F = '_'.join((RESULT_F, datetime.datetime.now().strftime('%Y-%m-%d'), '.csv'))


def main():
    mode = sys.argv[1]
    test_file = sys.argv[2]
    dic = {
        'train': lambda: pipeline_train(train_f=sys.argv[4], test_f=test_file, model_f=MODEL_F, result_f=RESULT_F,
                                        hdf_f=HDF_F,
                                        hdf_key=HDF_KEY, feature_conf=FEATURE_CONF, window_size=WINDOW_SIZE,
                                        col_names=HEADER),
        'validate': lambda: pipeline_validate(valid_f=VALIDATE_F, model_f=MODEL_F, result_f=RESULT_F, hdf_f=HDF_F,
                                              hdf_key=HDF_KEY, feature_conf=FEATURE_CONF, window_size=WINDOW_SIZE,
                                              col_names=HEADER),
        'chunk': lambda: pipline_predict(test_f=test_file, model_f=MODEL_F, hdf_f=HDF_F, hdf_key=HDF_KEY,
                                         feature_conf=FEATURE_CONF, window_size=WINDOW_SIZE)
    }
    dic[mode]()
