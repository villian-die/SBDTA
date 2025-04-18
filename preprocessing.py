import os.path as osp
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm
from collections import defaultdict

fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

def seqs2int(target):
    newsequence=''
    for i in range(len(target)):
        newsequence += dicts[target[i]]
    return [seq_dic2[s] for s in newsequence] 

dicts = {"H":"A","R":"A","K":"A",
         "D":"B","E":"B","N":"B","Q":"B",
         "C":"C","X":"C",
         "S":"D","T":"D","P":"D","A":"D","G":"D","U":"D",
         "M":"E","I":"E","L":"E","V":"E",
         "F":"F","Y":"F","W":"F"}

seq_rdic2 = ['A','B','C','D','E','F']

seq_dic2 = {w: i+1 for i,w in enumerate(seq_rdic2)}

def seq_cat(prot):
    x = np.zeros(1200)
    for i, ch in enumerate(prot[:1200]): 
        ch = dicts[ch]
        x[i] = seq_dic2[ch]
    return x 

class GNNDataset(InMemoryDataset):
    def __init__(self, root, Type ="train" , transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)     
        if Type == "train":
            self.data, self.slices = torch.load(self.processed_paths[0])
        if Type == "test":
            self.data, self.slices = torch.load(self.processed_paths[1])
        if Type == "val":
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_test.csv','data_val.csv']
    
    @property
    def processed_file_names(self):
        return ['processed_data_train.pt', 'processed_data_test.pt','processed_data_val.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def split_sequence2(sequence, ngram):
        sequence = '-' + sequence + '='
        words = [word_dict[sequence[i:i+ngram]]
                for i in range(len(sequence)-ngram+1)]
        return np.array(words)

    def process_data(self,xd, xt, y,smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        
        for i in range(data_len):
            smi = xd[i]
            sequence = xt[i]
            label = y[i]
            if smi in smile_graph:
                x, edge_index,edge_att = smile_graph[smi]
                protein = ''
                for i in sequence:     
                    i = int(i)
                    protein += str(i)
                ngram=3
                protein = '-' + protein+ '='
                words = [word_dict[protein[i:i+ngram]]
                         for i in range(len(protein)-ngram+1)]
                protein_AnJiSuan = np.array(words)
                data = DATA.Data(x=torch.Tensor(x),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([label]))
                data.protein_feature = torch.LongTensor([sequence])
                data.protein_AnJiSuan = torch.LongTensor([protein_AnJiSuan])
                data_list.append(data)
        return data_list
       
    def process(self):
        df_train = pd.read_csv(self.raw_paths[0])
        df_test = pd.read_csv(self.raw_paths[1])
        df_val = pd.read_csv(self.raw_paths[2])
        df = pd.concat([df_train, df_test])
        smiles = df['compound_iso_smiles'].unique()

        train_drugs, train_prots,  train_Y = list(df_train['compound_iso_smiles']),list(df_train['target_sequence']),list(df_train['affinity'])
        XT = [seq_cat(t) for t in train_prots]
        train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)
        
        test_drugs, test_prots,  test_Y = list(df_test['compound_iso_smiles']),list(df_test['target_sequence']),list(df_test['affinity'])
        XT = [seq_cat(t) for t in test_prots]
        test_drugs, test_prots,  test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)
         
        val_drugs, val_prots,  val_Y = list(df_val['compound_iso_smiles']),list(df_val['target_sequence']),list(df_val['affinity'])
        XT = [seq_cat(t) for t in val_prots]
        val_drugs, val_prots,  val_Y = np.asarray(val_drugs), np.asarray(XT), np.asarray(val_Y)
     
        smile_graph = {}
        for smile in smiles:
            g = self.mol2graph(smile,Max_node = 290)
            smile_graph[smile] = g

        train_list=self.process_data(xd=train_drugs,xt=train_prots,y=train_Y,smile_graph=smile_graph)
        test_list = self.process_data(xd=test_drugs, xt=test_prots, y=test_Y,smile_graph=smile_graph)
        val_list = self.process_data(xd=val_drugs, xt=val_prots, y=val_Y,smile_graph=smile_graph)
       
        if self.pre_filter is not None:
            train_list = [train for train in train_list if self.pre_filter(train)]
            test_list = [test for test in test_list if self.pre_filter(test)]
            val_list = [val for val in val_list if self.pre_filter(val)]

        if self.pre_transform is not None:
            train_list = [self.pre_transform(train) for train in train_list]
            test_list = [self.pre_transform(test) for test in test_list]
            val_list = [self.pre_transform(val) for val in val_list]
        print('Graph construction done. Saving to file.')

        data, slices = self.collate(train_list)
        torch.save((data, slices), self.processed_paths[0])       
        data, slices = self.collate(test_list)
        torch.save((data, slices), self.processed_paths[1])
        data, slices = self.collate(val_list)
        torch.save((data, slices), self.processed_paths[2])
     
    def get_nodes(self, g , Max_node):
        feat = []
        for n, d in g.nodes(data=True):
            h_t = []
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I', ]]
            h_t.append(d['a_num'])
            h_t.append(d['acceptor'])
            h_t.append(d['donor'])
            h_t.append(int(d['aromatic']))
            h_t += [int(d['hybridization'] == x) \
                    for x in (Chem.rdchem.HybridizationType.SP, \
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            h_t.append(d['ExplicitValence'])
            h_t.append(d['FormalCharge'])
            h_t.append(d['ImplicitValence'])
            h_t.append(d['NumExplicitHs'])
            h_t.append(d['NumRadicalElectrons'])
            feat.append((n, h_t))
        true_node = len(feat)
        virtual_list = []
        for i in range(22):
            virtual_list.append(0)
        for i in range(true_node,Max_node):
            feat.append((i, virtual_list))         
        feat.sort(key=lambda item: item[0])  
        node_attr = [item[1] for item in feat]
        return node_attr

    def get_edges(self, g):
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                   for x in (Chem.rdchem.BondType.SINGLE, \
                             Chem.rdchem.BondType.DOUBLE, \
                             Chem.rdchem.BondType.TRIPLE, \
                             Chem.rdchem.BondType.AROMATIC)]
            e_t.append(int(d['IsConjugated'] == False))
            e_t.append(int(d['IsConjugated'] == True))
            e[(n1, n2)] = e_t
        if len(e) == 0:
            return torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])
        edge_index = list(e.keys())
        edge_attr = list(e.values())
        return edge_index, edge_attr

    def mol2graph(self, smile,Max_node=50):
        self.Max_node = Max_node
        mol = Chem.MolFromSmiles(smile)
        feats = chem_feature_factory.GetFeaturesForMol(mol)
        g = nx.DiGraph()
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i,
                       a_type=atom_i.GetSymbol(),
                       a_num=atom_i.GetAtomicNum(),
                       acceptor=0,
                       donor=0,
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(),
                       ExplicitValence=atom_i.GetExplicitValence(),
                       FormalCharge=atom_i.GetFormalCharge(),
                       ImplicitValence=atom_i.GetImplicitValence(),
                       NumExplicitHs=atom_i.GetNumExplicitHs(),
                       NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
                       )

        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['acceptor'] = 1

        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j,
                               b_type=e_ij.GetBondType(),
                               # 1 more edge features 2 dim
                               IsConjugated=int(e_ij.GetIsConjugated()),
                               )
        node_attr = self.get_nodes(g,self.Max_node)
        L = np.array(node_attr)
        edge_index, edge_attr = self.get_edges(g)
        return node_attr, edge_index, edge_attr
    
if __name__ == "__main__":
    word_dict = defaultdict(lambda: len(word_dict))
    GNNDataset('datasets/davis')
    GNNDataset('datasets/kiba')
    #GNNDataset('datasets/human')
    #GNNDataset('datasets/biosnap')
    #GNNDataset('datasets/view')