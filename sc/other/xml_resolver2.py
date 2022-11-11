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
import csv
from tqdm import tqdm
import numpy as np

def to_csv(XML_PATH, SAVE_PATH):

    assert os.path.exists(XML_PATH), f"XML Path ERROR, Got {XML_PATH}"
    if True:
        print("Reading XMl file: ", XML_PATH)
        with open(XML_PATH, 'r') as f:
            xmlstr = f.read()
        print("Parsing XML: ", XML_PATH)
        _dict = xmltodict.parse(xmlstr)
    else:
        _dict = np.load("xml_dict.npy", allow_pickle=True)
        _dict = _dict.item()
    num_not_bug = 0
    ret_list = []
    for report in tqdm(_dict['ichicsr']['safetyreport']):
        rid = report['safetyreportid']
        onereport = dict()
        onereport['safetyreportid'] = rid
        reaction_list = []
        if isinstance(report['patient']['reaction'], dict):
            reaction = report['patient']['reaction']
            print("Buged : reportid: ", rid)
            if 'reactionoutcome' in reaction.keys() and reaction['reactionoutcome'] == '5':
                reaction_list.append(reaction)
        elif isinstance(report['patient']['reaction'], list):
            for reaction in report['patient']['reaction']:
                try:
                    if reaction['reactionoutcome'] == '5':
                        reaction_list.append(reaction)
                except:
                    pass
                finally:
                    pass
        else:
            raise ValueError(report)
        onereport['reaction'] = reaction_list
        if len(reaction_list) >0:
            ret_list.append(onereport)
            num_not_bug += 1
    print(len(ret_list))
    print("Not bug: ", num_not_bug)
    
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

    with open(SAVE_PATH, "w") as f:
        csv_writer = csv.writer(f)
        head = ['safetyreportid', 'reactionmeddraversionpt', 'reactionmeddrapt', 'reactionoutcome']
        csv_writer.writerow(head)
        for ret in ret_list:
            res = ret2list(ret)
            for r in res:
                csv_writer.writerow(r)


def to_csv2(XML_PATH, SAVE_PATH):

    assert os.path.exists(XML_PATH), f"XML Path ERROR, Got {XML_PATH}"
    reloading = True
    dname = "xml_dict.npy"
    if not reloading:
        print("Reading XMl file: ", XML_PATH)
        with open(XML_PATH, 'r') as f:
            xmlstr = f.read()
        print("Parsing XML: ", XML_PATH)
        _dict = xmltodict.parse(xmlstr)
        np.save(dname, _dict)
    else:
        print("Reloading")
        _dict = np.load(dname, allow_pickle=True)
        _dict = _dict.item()

    num_serious = 0
    num_not_bug = 0
    ret_list = []
    bug_list = []
    # import ipdb; ipdb.set_trace()
    for report in _dict['ichicsr']['safetyreport']:        
        # if "seriousnessdeath" in report.keys():
        if "seriousnessdeath" in report.keys() and report['seriousnessdeath'] == "1":
            num_serious += 1
            is_bug = False
            onereport = dict()
            rid = report['safetyreportid']
            onereport['seriousnessdeath'] = report['seriousnessdeath']
            onereport['safetyreportversion'] = report['safetyreportversion']
            onereport['safetyreportid'] = report['safetyreportid']
            # reaction_list = []
            # ret_list.append(onereport)
            # import ipdb; ipdb.set_trace()
            #  if isinstance(report['patient']['reaction'], dict):
            #    bug_list.append(rid)
            #    is_bug = True
            #    continue
            #for reaction in report['patient']['reaction']:
            #    if 'reactionoutcome' not in reaction.keys():
            #        is_bug = True
            #        continue
            #    if reaction['reactionoutcome'] != '5':
            #        is_bug = True
            #        continue
            #if is_bug:
            #    bug_list.append(rid)
            #    continue
            # num_not_bug += 1
            #     try:
            #         if reaction['reactionoutcome'] == '5':
            #             reaction_list.append(reaction)
            #     except:
            #         pass
            #     finally:
            #         pass
            # onereport['reaction'] = reaction_list
            # if len(reaction_list) > 0:
            #    # ret_list.append(onereport)
            onereport['reaction'] = report['patient']['reaction']
            ret_list.append(onereport)

            # import ipdb; ipdb.set_trace()
    print(len(ret_list))
    np.save("bug_list.npy", bug_list)
    print("Bug_list: ", bug_list)
    print("WTF? ", len(bug_list))
    print("num_serious: ", num_serious)
    print("No bug num: ", num_not_bug)
    
    def to_reaction(reaction, r):
        content = r
        if True:
            if isinstance(reaction, dict):
                s = []
                s += [r['seriousnessdeath']]
                s += [r['safetyreportversion']]
                s += [int(r['safetyreportid'])]
                if 'reactionmeddraversionpt' not in reaction.keys() or 'reactionmeddrapt' not in reaction.keys():
                    print("Reaction without reactionmeddrapt? ", reaction)
                    import ipdb; ipdb.set_trace()
                s += [reaction['reactionmeddraversionpt']] if 'reactionmeddraversionpt' in reaction.keys() else ["None"]
                s += [reaction['reactionmeddrapt']] if 'reactionmeddrapt' in reaction.keys() else ["None"]
                # assert isinstance(reaction['reactionoutcome'], str), f"{reaction}"
                s += [int(reaction['reactionoutcome'])]  if 'reactionoutcome' in reaction.keys() else [-1]
                    
                # alls.append(s)
            else:
                s = []
                s += [r['seriousnessdeath']]
                s += [r['safetyreportversion']]
                s += [int(r['safetyreportid'])]
                # alls.append(s)
                print("Debug: should be noticed: ", s, "\n --  reaction: ", reaction)
                import ipdb; ipdb.set_trace()
            for item in s:
                assert not isinstance(item, (dict, list)), s
            print("Debug: ", s)
            # import ipdb; ipdb.set_trace()
        return s

    def ret2list(r):
        content = r

        alls = []
        if not isinstance(r, dict):
            print("Debug r.dtype: ", type(r))
            print("content: ", r)
        if isinstance(r['reaction'], dict):
            reaction = r['reaction']
            alls.append(to_reaction(reaction, r))
            return alls
        assert isinstance(r['reaction'], list), r
        for reaction in r['reaction']:
            alls.append(to_reaction(reaction, r))
            #            import ipdb; ipdb.set_trace()
        return alls
    # np.save("alls.npy", alls)

    with open(SAVE_PATH, "w") as f:
        csv_writer = csv.writer(f)
        head = ['seriousnessdeath', 'safetyreportversion', 'safetyreportid', 'reactionmeddraversionpt', 'reactionmeddrapt', 'reactionoutcome']
        csv_writer.writerow(head)

        # ret_list = ret2list(ret_list)
        for ret in ret_list:
            ret = ret2list(ret)
            #if isinstance(ret, dict):
            #    ret = ret.values()
            #elif isinstance(ret, list):
            #    ret = ret
            #elif isinstance(ret, str):
            #    ret = [ret]
            #else:
            #    raise ValueError(f"ret bug {ret}")
            assert isinstance(ret, list), ret
            for item in ret:
                csv_writer.writerow(item)
            # res = ret2list(ret)
            # for r in res:
                # csv_writer.writerow(r)


def resolve_dir():
    xml_dir = './xml/'
    save_dir = './to_csv/'

    for path in tqdm(os.scandir(xml_dir)):
        path = path.path
        print("[*] Debug! , we are running on ", path)
        if path.endswith(".xml"):
            assert path.endswith(".xml"), f"Got Error path: {path}"
            parent, name = os.path.split(path)
            save_name = name.replace(".xml", "_converted.csv")
            to_csv(path, os.path.join(save_dir, save_name))
        print("[*] Finished ", path)

    
def resolve_file():
    filename = "/home1/quanquan/1_ADR20Q3.xml"
    save_name = "xxzhou2.csv"
    to_csv(filename, save_name)


if __name__ == '__main__':
    resolve_file()
