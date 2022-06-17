import pandas as pd
from utils import *


# check correctness groups
correct_groups = ["X1", "X2", "X3","X4","X5",
                  "V1", "V2", "V3", "V4", "V5", "V6", 
                  "S1", "S2", "S3", "S4", "S5", "S6"]
correct_groups_count = [1, 19, 1, 19, 29,
                        1, 15, 743, 15, 743, 1415,
                        1, 15, 743, 15, 743, 1415]
correct_groups_series = pd.Series(correct_groups_count, correct_groups)


class Dataset:
    full_structure = None
    structure_graph = None
    
    def __init__(self, data):
        self.data = data[['energy_per_atom', 
                          'initial_structure', 
                          'defect_representation']].copy()
        
        s0 = data.iloc[0].initial_structure
        
        self.layers_coords = layers_coords = build_layers_coords(s0)

        self.data['subgroup'] = self.data.defect_representation.apply(
            lambda x: classify(x, layers_coords)
        )
        self.data['group'] = self.data.subgroup.apply(lambda x: x[:2])
        
        pd.testing.assert_series_equal(correct_groups_series.sort_index(),
                               self.data.group.value_counts().sort_index(),
                               check_names=False)
        self.data['idx'] = np.arange(len(data))
        self.data.set_index('idx', inplace=True)
        
    def __getitem__(self, idx_or_name):
        if type(idx_or_name) == int:
            return self.data.iloc[idx_or_name]
        elif type(idx_or_name) == str:
            if idx_or_name in correct_groups:
                return self.data[self.data.group == idx_or_name]
            else:
                return self.data[self.data.subgroup == idx_or_name]
        raise ValueError

    
    def get_group_df(self, name):
        return self.data[self.data.group == name]
    
    def get_subgroup_df(self, name):
        return self.data[self.data.subgroup == name]
    
    def group_energy_argmin_idx(self, name):
        group_df = self.get_group_df(name)
        return group_df.iloc[group_df.energy_per_atom.argmin()].idx
    
    def subgroup_energy_argmin_idx(self, name):
        subgroup_df = self.get_subgroup_df(name)
        return subgroup_df.iloc[subgroup_df.energy_per_atom.argmin()].idx
        
    def get_full_structure(self):
        if self.full_structure is not None:
            return self.full_structure
        s0 = self.data.iloc[0].initial_structure
        d0 = self.data.iloc[0].defect_representation
        
        site_species = "S"
        defect_site = d0[0]
        insert_transf = InsertSitesTransformation([site_species],
                                                  [defect_site.frac_coords])
        self.full_structure = insert_transf.apply_transformation(s0)
        return self.full_structure
    
    def get_structure_graph(self):
        if self.structure_graph is not None:
            return self.structure_graph
        else:
            self.structure_graph = make_graph(self.get_full_structure())
            return self.structure_graph