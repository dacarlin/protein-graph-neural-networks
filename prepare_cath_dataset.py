import prody 
from random import shuffle, seed
import json 
from Bio.SeqUtils import IUPACData
from rich.progress import track 
import pandas 


seed(42)

data_list = "data/cath-dataset-nonredundant-S40.list"
splits_json = "data/splits.json"


with open(data_list) as fn:
    files = list(x.strip() for x in fn.readlines()) 

len(files)

shuffle(files)  # respects `seed`, above 

n1 = int(0.8 * len(files))
n2 = int(0.9 * len(files))

train = files[0:n1]
val = files[n1:n2]
test = files[n2:]

print(f"{len(train)=}, {len(val)=}, {len(test)=}")

assert "2imiB02" == train[0]  # check that the seed is working 

splits = {
    "train": train, 
    "val": val, 
    "test": test, 
}

with open(splits_json, "w") as fn:
    json.dump(splits, fn, indent=2)

max_samples = 50_000 
datapoints = []

for split, values in splits.items():
    valid = 0 
    invalid = 0 

    for value in track(values[:max_samples], description=split):
        try:
            path = f"data/dompdb/{value}"
            with open(path) as fn:
                atom_group = prody.parsePDBStream(fn)

            sequence = ""
            xyz = [] 
            beta = []

            for residue in atom_group.iterResidues():
                try:
                    xyz_stack = (
                        residue.getAtom("N").getCoords().tolist(),
                        residue.getAtom("CA").getCoords().tolist(),
                        residue.getAtom("C").getCoords().tolist(),
                        residue.getAtom("O").getCoords().tolist(),
                    )
                except:
                    xyz_stack = (
                        [np.nan, np.nan, np.nan], 
                        [np.nan, np.nan, np.nan], 
                        [np.nan, np.nan, np.nan], 
                        [np.nan, np.nan, np.nan], 
                    )
                name3 = residue.getResname() 
                name1 = IUPACData.protein_letters_3to1[name3.capitalize()]
                num = residue.getResnum() 
                betas = residue.getBetas() 
                sum_betas = sum(betas)

                sequence += name1 
                xyz.append(xyz_stack)
                beta.append(sum_betas)

            #xyz = np.stack(xyz)
            valid += 1

            assert len(xyz) == len(sequence)
            
            datapoints.append({
                "sequence": sequence, 
                "xyz": xyz, 
                "name": value, 
                "split": split, 
            })
        except:
            invalid += 1 

    print(f"split={split}, valid={valid}, invalid={invalid} ({invalid/(valid+invalid)*100}%)")
 


print(datapoints[69])
print(datapoints[66])


with open("data/chains.jsonl", "w") as fn:
    for sample in datapoints:
        fn.write(json.dumps(sample) + "\n")

# calculate statistics 

data = []
with open("data/chains.jsonl") as fn:
    for line in fn.readlines():
        pkg = json.loads(line)
        del pkg["xyz"]
        data.append(pkg)

df = pandas.DataFrame(data)
df["len"] = df["sequence"].map(lambda x: len(x))
df.to_csv("data/summary.csv")

print(df.head())

for split, my_df in df.groupby(by="split"):
    # num_samples 
    num_samples = len(my_df)
    min_length = my_df["len"].min() 
    max_length = my_df["len"].max() 
    mean_length = my_df["len"].mean() 

    print(f"{split=} {num_samples=} {min_length=} {mean_length=:.0f} {max_length=}")


    



