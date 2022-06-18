from os import walk

alreadycalculated = []

for path,subirs,files in walk("data"):
  for name in files:
    alreadycalculated.append(name.split(".")[0])

print(alreadycalculated)
