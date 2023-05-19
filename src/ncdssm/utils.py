from ncdssm.type import Dict


def listofdict2dictoflist(lod: Dict):
    keys = lod[0].keys()
    return {k: [elem[k] for elem in lod] for k in keys}
