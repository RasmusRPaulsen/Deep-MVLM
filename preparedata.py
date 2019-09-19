import argparse
from parse_config import ConfigParser
import os
import socket


def create_lock_file(name):
    f = open(name, "w")
    f.write(socket.gethostname())
    f.close()


def delete_lock_file(name):
    if os.path.exists(name):
        os.remove(name)


def process_file_bu_3dfe(config, file_name, o_dir):
    bu_3dfe_dir = config['preparedata']['raw_data_dir']
    base_name = os.path.basename(file_name)
    name_pd = bu_3dfe_dir + file_name + '_RAW.wrl'
    name_bmp = bu_3dfe_dir + file_name + '_F3D.bmp'
    name_lm =  bu_3dfe_dir + file_name + '_RAW_84_LMS.txt'
    lock_file = o_dir + base_name + '.lock'

    if not os.path.isfile(name_pd):
        print(name_pd, ' could not read')
        return False
    if not os.path.isfile(name_bmp):
        print(name_bmp, ' could not read')
        return False
    if not os.path.isfile(name_lm):
        print(name_lm, ' could not read')
        return False
    if os.path.isfile(lock_file):
        print(file_name, ' is locked - skipping')
        return True
    create_lock_file(lock_file)



    delete_lock_file(lock_file)

def prepare_bu_3dfe_data(config):
    print('Preparing BU-3DFE data')
    file_id_list = config['preparedata']['raw_data_dir'] + 'BU_3DFE_base_filelist_noproblems.txt'
    output_dir = config['preparedata']['processed_data_dir']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_file_names = []
    with open(file_id_list) as f:
        for line in f:
            line = line.strip("/n")
            line = line.strip("\n")
            if len(line) > 0:
                base_file_names.append(line)
    print('Read ', len(base_file_names), ' file ids')

    for base_name in base_file_names:
        print('Processing ', base_name)
        name_path = os.path.dirname(base_name)
        o_dir = output_dir + name_path + '/'
        if not os.path.exists(o_dir):
            os.makedirs(o_dir)
        process_file_bu_3dfe(config, base_name, o_dir)


def main(config):
    model_data_name = config['name']
    if model_data_name.find('BU_3DFE') > -1:
        prepare_bu_3dfe_data(config)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Deep-MVLM')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    cfg_global = ConfigParser(args)
    main(cfg_global)
