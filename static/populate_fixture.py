#!/usr/bin/env python3
import json
import sys

def create_fixture(raw_data_file):
    with open(raw_data_file, "r") as raw_json:
        tech_data = json.load(raw_json)
        ioi_to_tbl = [("Equipamento_Tipo", "Equipamento_Tipo"),
                      ("Servico", "ServiceType"),
                      ("Tarifario", "Tarifario")]
        fixture_data = []
        for ioi, db_tbl in ioi_to_tbl:
            for entry in tech_data[ioi]:
                tmp = { "model": "technical_problems." + db_tbl,
                        "fields": {
                            "name": entry
                        }
                }
                fixture_data.append(tmp)

    with open("input_options_fixture.json", "w") as dump:
        json.dump(fixture_data, dump, indent=2)

if __name__=="__main__":
    if(len(sys.argv) > 1):
        create_fixture(sys.argv[1])
