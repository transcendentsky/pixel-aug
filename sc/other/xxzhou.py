"""
Resolve Data from XX.Zhou
<ichicsr >
    <ichicsrmessageheader>
        ..
    </>
    <safetyreport>
        <safetyreportid>1023435</>
        <reaction>
            <reactionmeddrapt> Hyponatrdas </>
            <reactionoutcome>5</>
        </>
</>
"""
#
# import xml.etree.ElementTree as ET
# tree = ET.parse('./demo.xml')
# print(tree)
# root = tree.getroot()
# for child in root:
#     print(child.tag, child.attrib)

import xmltodict
import json
import os


def ret2list(r):
    alls = []
    for reaction in r['reaction']:
        s = []
        s += [int(r['safetyreportid'])]
        s += [reaction['reactionmeddraversionpt']]
        s += [reaction['reactionmeddrapt']]
        s += [int(reaction['reactionoutcome'])]
        alls.append(s)
    return alls

def to_csv(XML_PATH, SAVE_PATH):

    assert os.path.exists(XML_PATH), f"XML Path ERROR, Got {XML_PATH}"

    with open(XML_PATH, 'r') as f:
        xmlstr = f.read()
    _dict = xmltodict.parse(xmlstr)

    ret_list = []
    for report in _dict['ichicsr']['safetyreport']:
        rid = report['safetyreportid']
        onereport = dict()
        onereport['safetyreportid'] = rid
        reaction_list = []
        for reaction in report['patient']['reaction']:
            try:
                if reaction['reactionoutcome'] == '5':
                    reaction_list.append(reaction)
            except:
                pass
            finally:
                pass
        onereport['reaction'] = reaction_list
        if len(reaction_list) >0:
            ret_list.append(onereport)
    print(len(ret_list))



    import csv
    with open(SAVE_PATH, "w") as f:
        csv_writer = csv.writer(f)
        head = ['safetyreportid', 'reactionmeddraversionpt', 'reactionmeddrapt', 'reactionoutcome']
        csv_writer.writerow(head)
        for ret in ret_list:
            res = ret2list(ret)
            for r in res:
                csv_writer.writerow(r)


if __name__ == '__main__':

    xml_dir = './xml/'
    save_dir = './to_csv/'

    for path in os.scandir(xml_dir):
        parent, name = os.path.split()
        save_name = name.replace(".xml", "_converted.csv")
        to_csv(path, os.path.join(save_dir, save_name))
