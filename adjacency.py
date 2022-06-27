from utils import *
from joblib import Parallel, delayed

class MatrixBuilder:
    def __init__(self, dataset, name):
        self.name = name
        self.group_df = dataset[name].copy()
        self.group_df['distance_graph'] = self.group_df.defect_representation.apply(
            lambda x: nx.from_numpy_matrix(x.distance_matrix)
        )
        self.full_structure = dataset.get_full_structure()
        finder = SpacegroupAnalyzer(self.full_structure,
                                    symprec=1e-1)
        self.checker = finder.get_space_group_operations()
        self.metrics = {"n_symmetry_checks" : 0,
                        "n_find_sym_idx": 0,
                        "sum_len_to_search": 0}
#         self.structure_graph = dataset.get_structure_graph()
#         self.group_df["subgraph"] = self.group_df.defect_representation.apply(
#             lambda x: make_subgraph(self.structure_graph, 
#                                     self.full_structure, 
#                                     x))
        
    def reset_metrics(self):
        self.metrics = {"n_symmetry_checks" : 0,
                        "n_find_sym_idx": 0,
                        "sum_len_to_search": 0}
        
    def swap_sites(self, structure, defect_site, idx):
        """
        Swap defect site with site in structure by index.
        Works correctly when defect is substitution or vacancy
        """
        defect_species = defect_site.species_string
        site_species = structure[idx].species_string
        if defect_species == site_species:
            return structure.copy(), defect_site
        new_defect_site = PeriodicSite(defect_site.species, 
                                       structure[idx].frac_coords, 
                                       structure[idx].lattice)
        if defect_species != "X0+":
            defect_idx = find_site(structure, defect_site)
            indices_species_map = {idx: defect_species, 
                                   defect_idx: site_species}
            transformation = ReplaceSiteSpeciesTransformation(indices_species_map)
            return transformation.apply_transformation(structure), new_defect_site
        insert_transf = InsertSitesTransformation([site_species],
                                                  [defect_site.frac_coords])
        remove_transf = RemoveSitesTransformation([idx])
        removed_s = remove_transf.apply_transformation(structure)
        return insert_transf.apply_transformation(removed_s), new_defect_site
    
    def swap_only_defect_sites(self, structure, defect_site, idx):
        """
        Swap defect site with site in structure by index.
        Works correctly when defect is substitution or vacancy
        """
        defect_species = defect_site.species_string
        site_species = structure[idx].species_string
        if defect_species == site_species:
            return None, defect_site
        new_defect_site = PeriodicSite(defect_site.species, 
                                       structure[idx].frac_coords, 
                                       structure[idx].lattice)
        if defect_species != "X0+":
            defect_idx = find_site(structure, defect_site)
            indices_species_map = {idx: defect_species, 
                                   defect_idx: site_species}
#             transformation = ReplaceSiteSpeciesTransformation(indices_species_map)
            return None, new_defect_site
#         insert_transf = InsertSitesTransformation([site_species],
#                                                   [defect_site.frac_coords])
#         remove_transf = RemoveSitesTransformation([idx])
#         removed_s = remove_transf.apply_transformation(structure)
        return None, new_defect_site
    
    
    def generate_neighbouring_structures(self, structure, defects, how=None):
        """
        Only one swap.
        """
        def defects_are_eq(d1, d2):
            if how is None:
                return self.checker.are_symmetrically_equivalent(d1, 
                                                            d2, 
                                                            symm_prec=0.01)
            elif how == "DG":
                if not matrices_are_isomofic(d1.distance_matrix, 
                                             d2.distance_matrix):
                    return False
                else:
                    return self.checker.are_symmetrically_equivalent(d1, 
                                                                d2, 
                                                                symm_prec=0.01)
            elif how == "SG":
                SG1 = make_subgraph(self.structure_graph, 
                           self.full_structure,
                           d1)
                SG2 = make_subgraph(self.structure_graph, 
                           self.full_structure,
                           d2)
                if not nx.is_isomorphic(SG1, SG2):
                    return False
                else:
                    return self.checker.are_symmetrically_equivalent(d1, 
                                                                d2, 
                                                                symm_prec=0.01)

            raise ValueError(f"Undefined comparison style: {how}")
            
        result_structures = []
        defect_reprs = []
        for i, defect_site in enumerate(defects):
            nns = get_nn(structure, defect_site)
            for n_idx in nns:
                new_structure, new_defect_site = self.swap_only_defect_sites(structure, 
                                                                             defect_site, 
                                                                             n_idx)
                new_defect_repr = defects.copy()
                new_defect_repr[i] = new_defect_site

                # check if swapped with another defect
                found, idx = safe_find_site(defects, new_defect_site)
                if found:
                    swapped_defect_site = PeriodicSite(defects[idx].species, 
                                       defect_site.frac_coords, 
                                       defect_site.lattice)
                    new_defect_repr[idx] = swapped_defect_site
                # check if got something different from 
                # initial conf and all generated before
                is_new = True
                for reprn in defect_reprs+[defects]:
                    if defects_are_eq(reprn, new_defect_repr):
                        is_new = False
                        break
                if is_new or len(defect_reprs) == 0:
                    result_structures.append(new_structure)
                    defect_reprs.append(new_defect_repr)
        return result_structures, defect_reprs
    
    def get_similar_by_dg(self, defect_representation, rtol=1.5e-2):
        DG = nx.from_numpy_matrix(defect_representation.distance_matrix)
        edge_match = iso.numerical_edge_match("weight", 0.0, rtol=rtol)
        match = lambda x: nx.is_isomorphic(x, DG, edge_match=edge_match)
        return self.group_df[np.vectorize(match)(
            self.group_df.distance_graph
        )]

    
    def get_similar_by_sg(self, defect_representation, rtol=0.5):
        SG = make_subgraph(self.structure_graph, 
                           self.full_structure,
                           defect_representation)
        match = lambda x: nx.is_isomorphic(x, SG)
        return self.group_df[np.vectorize(match)(
            self.group_df.subgraph
        )]
    
    def get_equivalent_idx(self, defect_representation, how=None):
        self.metrics["n_find_sym_idx"] += 1
        if how is None:
            to_search = self.group_df
        elif how=="DG":
            to_search = self.get_similar_by_dg(defect_representation)
        elif how=="SG":
            to_search = self.get_similar_by_sg(defect_representation)
        self.metrics["sum_len_to_search"] += len(to_search)
        if len(to_search) == 1:
            return int(to_search.index[0])
        for i, row in to_search.iterrows():
            self.metrics["n_symmetry_checks"] += 1 
            if self.checker.are_symmetrically_equivalent(defect_representation,
                                                    row.defect_representation,
                                                    symm_prec=0.01):
                return i
        raise ValueError(
            f"Not found symmetrically equivalent structure for {defect_representation}"
        )
        
    def _generate_delta_e_dict(self, how=None, n_jobs=1):
        """
        Returns dict with E_dst-E_src (energies of destination and source conf)
        """
        def get_item(i, row):
            adj_dict = {}
            cur_energy = row.energy_per_atom
            ns, defect_reprs = self.generate_neighbouring_structures(
                row.initial_structure, row.defect_representation, how=how)
            adj_dict[i] = {}
            for defect_repr in defect_reprs:
                idx = self.get_equivalent_idx(defect_repr, how)
                energy = self.group_df.loc[idx].formation_energy
                if idx != i:
                    adj_dict[i] |= {idx: {"weight": energy - cur_energy}}
            return adj_dict

        result = Parallel(n_jobs=n_jobs, verbose=1000)(
            delayed(get_item)(i, row) for i, row in self.group_df.iterrows()
        )
        adj_dict = {}
        for entry in result:
            adj_dict |= entry
        return adj_dict
        
        
    def generate_group_adj_dict(self, thr=0.0, n_jobs=1, how=None):
        """
        Parallel
        """
        def get_item(i, row):
            adj_dict = {}
            cur_energy = row.formation_energy
            ns, defect_reprs = self.generate_neighbouring_structures(
                row.initial_structure, row.defect_representation, how=how)
            adj_dict[i] = []
            for defect_repr in defect_reprs:
                idx = self.get_equivalent_idx(defect_repr, how)
                energy = self.group_df.loc[idx].formation_energy
                if energy - cur_energy <= thr and idx != i:
                    adj_dict[i].append(idx)
            return adj_dict

        result = Parallel(n_jobs=n_jobs, verbose=1000)(
            delayed(get_item)(i, row) for i, row in self.group_df.iterrows()
        )
        adj_dict = {}
        for entry in result:
            adj_dict |= entry
        return adj_dict