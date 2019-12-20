#!/usr/bin/env python3
import json
import sys
import random

def getMatch(desc, ss):
    return ss[0], random.uniform(0,1)


if __name__=='__main__':
    if len(sys.argv) == 3:
        mia = {
            'Sintoma': ('', 0.0),
            'Tipificacao_Nivel_1':  ('', 0.0),
            'Tipificacao_Nivel_2':  ('', 0.0),
            'Tipificacao_Nivel_3':  ('', 0.0),
        }

        service = sys.argv[1]
        desc = sys.argv[2]
        
        with open('tree_options.json', 'r') as stj:
            st = json.load(stj)
        
        cs = st['Servico'][service]
        for ia in mia:
            cs = cs[ia]
            ss = [st for st in cs]
            mt,pb = getMatch(desc, ss)
            if pb < 0.65:
                #TODO: request more info OR return support contacts
                break
            mia[ia] = (mt,pb)
            if ia != 'Tipificacao_Nivel_3':
                cs = cs[mt]
        print(mia)
    else:
        print('Usage: ./{} <service> <description>'.format(sys.argv[0]))
    
    
