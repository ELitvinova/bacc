import numpy as np
import nglview
import json
import tempfile
import networkx as nx
import networkx.algorithms.isomorphism as iso

def get_tmp_filename():
    return next(tempfile._get_candidate_names())

from pymatgen.transformations.site_transformations import (
    ReplaceSiteSpeciesTransformation,
    InsertSitesTransformation,
    RemoveSitesTransformation
)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SpacegroupOperations
from pymatgen.core.sites import PeriodicSite

from pymatgen.core import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN


def get_element(defect_site):
    return defect_site.as_dict()["species"][0]["element"]

def get_layer(defect_site, layers_coords, atol=1e-02):
    """
    layers_coords - array/list of len 3
    output:
    -1 - S (bottom)
     0 - Mo
     1 - S (top)
    """
    z = defect_site.frac_coords[2]
    is_close = np.isclose(layers_coords, z, atol)
    assert np.max(is_close) != 0
    
    return np.argmax(np.isclose(layers_coords, z, atol)) - 1

def build_layers_coords(structure):
    zs = [site.frac_coords[2] for site in structure.sites]
    layer_coords = np.zeros(3)
    layer_coords[0], layer_coords[2] = np.min(zs), np.max(zs)
    layer_coords[1] = (layer_coords[0] + layer_coords[2]) / 2
    return layer_coords

def classify(defect_representation, layers_coords, atol=1e-02):
    """
    Detect subgroup to which defect representation belongs. Group+"same/diff"
    """
    def build_suffix_2_defects(elements, layers):
        """
        both arguments of len 2
        determine S-defects are on the same layer and which one
        """
        assert len(elements) == 2
        assert len(layers) == 2

        if layers[0] == layers[1]:
            return "same"
        return "diff" 
    
    def classify_X(elements, layers):
        number, suffix = "", ""
        if len(elements) == 1:
            number = 3 if elements[0] == "Se" else 1
        else:
            suffix = build_suffix_2_defects(elements, layers)
            if "Se" not in elements:
                number = 2
            elif "X" not in elements:
                number = 4         
            else: 
                number = 5
        return number, suffix

    def classify_V_or_S(elements, layers):
        number, suffix = "", ""
        if len(elements) == 1:
            number = 1
        else:
            mo_w_idx = layers.index(0)
            number, suffix = classify_X(elements[:mo_w_idx] + elements[mo_w_idx+1:],
                                        layers[:mo_w_idx] + layers[mo_w_idx+1:])
            number += 1
        return number, suffix
    
    elements = [get_element(site) for site in defect_representation.sites]
    layers = [get_layer(site, layers_coords) for site in defect_representation.sites]
    prefix = ""
    if 0 not in layers:
        prefix = "X"
        number, suffix = classify_X(elements, layers)
    elif "W" in elements:
        prefix = "S"
        number, suffix = classify_V_or_S(elements, layers)
    else:
        number,suffix = classify_V_or_S(elements, layers)
        prefix = "V"
    return f"{prefix}{number}_{suffix}" if suffix else f"{prefix}{number}"

# work with sites

def sites_equal(s1, s2, atol=1.5e-1):
    return np.allclose(s1.coords, s2.coords, atol=atol)

def find_site(structure, site, atol=1.5e-1):
    for i, s in enumerate(structure):
        if sites_equal(s, site, atol):
            return i
    raise ValueError(f"{site} not found in structure with atol={atol}")
    
def safe_find_site(sites, site, atol=1.5e-1):
    for i, s in enumerate(sites):
        if sites_equal(s, site, atol):
            return True, i
    return False, -1
    
def get_x_y_dist(structure, size=(8,8)):
    ls = structure.lattice.lengths
    return ls[0] / size[0], ls[1] / size[1]

def convert_site(site):
    if site.specie.symbol == "X":
        return PeriodicSite("Au", site.frac_coords, site.lattice)
    return site

def get_r(structure, size=(8,8,3)):
    ls = np.array(structure.lattice.lengths) / size
    return np.sqrt(np.sum(np.square(ls))) / 2

def matrices_are_isomofic(m1, m2, rtol=1e-2):
    if m1.shape != m2.shape:
        return False
    G1 = nx.from_numpy_matrix(m1)
    G2 = nx.from_numpy_matrix(m2)
    edge_comp = iso.numerical_edge_match("weight", 0.0, rtol=rtol)
    return nx.is_isomorphic(G1, G2, edge_match=edge_comp)

def get_nn(structure, site, atol=1e-1):
        """
        returns indicies of neighbouring sites in structure for given site.
        """
        layer_coords = build_layers_coords(structure)
        layer = get_layer(site, layer_coords)
        nearest_neighbor_ids = []
        dx, dy = get_x_y_dist(structure)
        for i,s in enumerate(structure):
            if max(np.isclose(site.distance(s), [dx, dy], atol=atol)) == 1 and get_layer(s, layer_coords) == layer:
                nearest_neighbor_ids.append(i)    
        return nearest_neighbor_ids
    
def make_graph(structure): 
    graph = StructureGraph.with_local_env_strategy(structure, 
                                                  MinimumDistanceNN(
                                                      cutoff=get_r(structure), 
                                                      get_all_sites=True),
                                                  weights=True
                                                  )
    graph.set_node_attributes()
    return graph

def get_sites_of_subgraph(structure_graph, full_structure, defect_representation):
    n = len(defect_representation)
    sites = set([convert_site(defect) for defect in defect_representation])
    for i in range(n):
        d_i = find_site(full_structure, defect_representation[i])
        for j in range(i+1, n, 1):
            d_j = find_site(full_structure, defect_representation[j])
            graph_ij = nx.Graph(structure_graph.graph)
            for k, d in enumerate(defect_representation):
                if d.specie.symbol == 'X' and k != i and k !=j :
                    graph_ij.remove_node(find_site(full_structure, d))
            path = nx.shortest_path(graph_ij, d_i, d_j)
            for idx in path[1:-1]:
                site_in_struct = full_structure[idx]
                is_found, index = safe_find_site(defect_representation,
                                                 site_in_struct)
                if not is_found:
                    sites.add(site_in_struct)
    return list(sites)

def make_subgraph(structure_graph, full_structure, defect_representation):
    sites = get_sites_of_subgraph(structure_graph, full_structure, defect_representation)
    struct = Structure.from_sites(sites)
    subgraph = nx.Graph(make_graph(struct).graph)
    idx = [find_site(struct, defect) for defect in defect_representation]

    return nx.algorithms.approximation.steiner_tree(subgraph, idx)
# plot structure

# https://gist.github.com/lan496/3f60b6474750a6fd2b4237e820fbfea4
def plot3d(structure, spacefill=True, show_axes=True):
    from itertools import product
    from pymatgen.core import Structure
    from pymatgen.core.sites import PeriodicSite
    
    eps = 1e-8
    sites = []
    for site in structure:
        species = site.species
        frac_coords = np.remainder(site.frac_coords, 1)
        for jimage in product([0, 1 - eps], repeat=3):
            new_frac_coords = frac_coords + np.array(jimage)
            if np.all(new_frac_coords < 1 + eps):
                new_site = PeriodicSite(species=species, coords=new_frac_coords, lattice=structure.lattice)
                sites.append(new_site)
    structure_display = Structure.from_sites(sites)
    
    view = nglview.show_pymatgen(structure_display)
    view.add_unitcell()
    
    if spacefill:
        view.add_spacefill(radius_type='vdw', radius=0.5, color_scheme='element')
        view.remove_ball_and_stick()
    else:
        view.add_ball_and_stick()
        
    if show_axes:
        view.shape.add_arrow([-4, -4, -4], [0, -4, -4], [1, 0, 0], 0.5, "x-axis")
        view.shape.add_arrow([-4, -4, -4], [-4, 0, -4], [0, 1, 0], 0.5, "y-axis")
        view.shape.add_arrow([-4, -4, -4], [-4, -4, 0], [0, 0, 1], 0.5, "z-axis")
        
    view.camera = "perspective"
    return view

# Save and load dicts

def save_adj_dict(adj_dict, name):
    with open(f"{name}.json", "w") as json_file:
        json.dump(adj_dict, json_file)
        
def restore_adj_dict(group_name):
    def jsonKeys2int(x):
        if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
        return x

    adj_dct = {}
    with open(f"{group_name}.json", "r") as json_file:
         adj_dct = json.load(json_file, object_hook=jsonKeys2int)
    return adj_dct