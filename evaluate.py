import torch
import numpy as np
import scipy.stats as stats
import argparse,os,json,h5py
from torch.utils.data import DataLoader
from planet.model import PLANET
from planet.data import ProLigDataset
from rdkit import RDLogger


def concordance_index(predicted, actual):
    """Concordance Index (CI): fraction of pairs where ranking is correct."""
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    pairs = 0
    concordant = 0.0
    for i in range(len(actual)):
        for j in range(i + 1, len(actual)):
            if actual[i] == actual[j]:
                continue
            pairs += 1
            diff_pred = predicted[i] - predicted[j]
            diff_act = actual[i] - actual[j]
            if diff_pred * diff_act > 0:
                concordant += 1.0
            elif diff_pred == 0:
                concordant += 0.5
    return concordant / pairs if pairs > 0 else 0.0


def evaluate(model, test_dataset):
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=4,drop_last=False,collate_fn=lambda x:x[0])
    bonded_pairs = test_dataset.get_bonded_atom_pairs()
    model.eval()
    predicted_lig_interactions,predicted_interactions,predicted_affinities,lig_scopes,res_scopes = [],[],[],[],[]
    ligand_interactions,pro_lig_interactions,pKs = [],[],[]

    with torch.no_grad():
        for (res_feature_batch,mol_feature_batch,targets) in test_loader:
            try:
                (predicted_lig_interaction,predicted_interaction,predicted_affinity) = model(res_feature_batch,mol_feature_batch)
                ligand_interaction,pro_lig_interaction,pK,_,_ = targets

                predicted_lig_interactions.append(np.array(predicted_lig_interaction.squeeze().detach().cpu()))
                predicted_interactions.append(np.array(predicted_interaction.squeeze().detach().cpu()))
                predicted_affinities.append(np.array(predicted_affinity.squeeze().detach().cpu()))
                lig_scopes.extend(mol_feature_batch[4])
                res_scopes.extend(res_feature_batch[2])

                ligand_interactions.append(np.array(ligand_interaction.squeeze().detach().cpu()))
                pro_lig_interactions.append(np.array(pro_lig_interaction.squeeze().detach().cpu()))
                pKs.append(np.array(pK.squeeze().detach().cpu()))

            except Exception as e:
                print(e)
                continue

    predicted_lig_interactions = np.concatenate(predicted_lig_interactions,axis=0)
    predicted_interactions = np.concatenate(predicted_interactions,axis=0)
    predicted_affinities = np.concatenate(predicted_affinities,axis=0)
    ligand_interactions = np.concatenate(ligand_interactions,axis=0)
    pro_lig_interactions = np.concatenate(pro_lig_interactions,axis=0)
    pKs = np.concatenate(pKs,axis=0)

    MAE = np.mean(np.abs(predicted_affinities-pKs))
    RMSE = np.sqrt(np.mean(np.square(predicted_affinities-pKs)))
    P_correlation, _ = stats.pearsonr(predicted_affinities, pKs)
    S_correlation, _ = stats.spearmanr(predicted_affinities, pKs)
    CI = concordance_index(predicted_affinities, pKs)
    print('MAE:{:.3f}\tRMSE:{:.3f}\tPearson R:{:.3f}\tSpearman:{:.3f}\tCI:{:.3f}'.format(
        MAE, RMSE, P_correlation, S_correlation, CI))

    return (predicted_lig_interactions, predicted_interactions, predicted_affinities,
            ligand_interactions, pro_lig_interactions, pKs, lig_scopes, res_scopes, bonded_pairs)


if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--model_file', required=True)
    parser.add_argument('-c','--casf_dir', required=True,
                        help='Directory with preprocessed pocket h5 files')
    parser.add_argument('-o','--out_path', required=True)
    parser.add_argument('--index', default=None,
                        help='PDBbind index file — evaluate only these PDB codes')

    parser.add_argument('--feature_dims', type=int, default=300)
    parser.add_argument('-n','--nheads', type=int, default=8)
    parser.add_argument('--key_dims', type=int, default=300)
    parser.add_argument('-va','--value_dims', type=int, default=300)
    parser.add_argument('-pu','--pro_update_inters', type=int, default=3)
    parser.add_argument('-lu','--lig_update_iters', type=int, default=10)
    parser.add_argument('-pl','--pro_lig_update_iters', type=int, default=1)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PLANET(args.feature_dims,args.nheads,args.key_dims,args.value_dims,
                   args.pro_update_inters,args.lig_update_iters,args.pro_lig_update_iters,device).to(device)
    model.load_state_dict(torch.load(args.model_file, map_location=device, weights_only=True))

    pdb_ids = None
    if args.index:
        pdb_ids = set()
        with open(args.index) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                pdb_ids.add(line.split()[0].lower())

    test_dataset = ProLigDataset(args.casf_dir, pdb_ids=pdb_ids, split='all',
                                 batch_size=16, shuffle=False, decoy_flag=False)

    (predicted_lig_interactions, predicted_interactions, predicted_affinities,
     ligand_interactions, pro_lig_interactions, pKs,
     lig_scopes, res_scopes, bonded_pairs) = evaluate(model, test_dataset)

    out_h5 = args.out_path if args.out_path.endswith('.h5') else args.out_path + '.h5'
    out_json = out_h5[:-3] + '_meta.json'
    with h5py.File(out_h5, 'w') as f:
        f.create_dataset('predicted_lig_interactions', data=predicted_lig_interactions)
        f.create_dataset('predicted_interactions',     data=predicted_interactions)
        f.create_dataset('predicted_affinities',       data=predicted_affinities)
        f.create_dataset('ligand_interactions',        data=ligand_interactions)
        f.create_dataset('pro_lig_interactions',       data=pro_lig_interactions)
        f.create_dataset('pKs',                        data=pKs)
    with open(out_json, 'w') as f:
        json.dump({'lig_scopes': lig_scopes, 'res_scopes': res_scopes,
                   'bonded_pairs': bonded_pairs}, f)
