from synthrl.language.bitvector.oracle import OracleSampler

if __name__ == '__main__':
    dataset = OracleSampler(1000,20,seed=None)
    dataset.to_json("./dataset.json")