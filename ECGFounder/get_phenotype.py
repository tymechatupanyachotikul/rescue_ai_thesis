import torch 

phenotypes = torch.load('/projects/prjs1252/data/UKBB/phenotype_targets.pt')
torch.save(phenotypes, '/projects/prjs1890/uk_biobank/phenotype_targets.pt')