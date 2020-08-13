from synthrl.language.bitvector.oracle import OracleSampler
from synthrl.utils.trainutils import Dataset
if __name__ == '__main__':
    # dataset = OracleSampler(10,5)
    dataset = Dataset.from_json("./dataset.json")
    print(dataset)