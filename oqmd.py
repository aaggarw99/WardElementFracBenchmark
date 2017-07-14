import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn import svm
import pylab
import seaborn as sns
import pymatgen as mg

data = pd.read_csv("data/oqmd_all.csv", delim_whitespace=True, na_values=['None'])
# graphs composition vs volume
def composition_vs_volume(comp, cut):
    count = cut
    isolation_var = comp

    comp = data.head(count).comp
    vol = data.head(count).volume_pa

    cut_comp = []
    cut_vol = []
    for i in range(len(comp)):
        if comp[i].find(isolation_var) != -1:
            cut_comp.append(comp[i])
            cut_vol.append(vol[i])

    index = np.arange(len(cut_comp))
    plt.scatter(index, cut_vol, color="black", label="Data")
    plt.xticks(index, cut_comp, rotation=90)
    plt.xlabel("Compositions with "+comp)
    plt.ylabel("Volume")
    plt.title("Compositions with "+isolation_var+" vs Volume PA")
    plt.legend()
    plt.show()

# composition_vs_volume('Li', 1000)

# graphs a strip plot of any number of compositions
def stripplot_comps(comps, cut):
    if isinstance(comps, list) == False:
        return

    count = cut
    total_comp = data.head(count).comp
    total_vol = data.head(count).volume_pa

    temp_c = []
    temp_v = []
    temp_cat = []

    cut_comp = []
    cut_vol = []
    cut_category = []
    for i in range(len(comps)):
        temp_c = []
        temp_v = []
        temp_cat = []
        for j in range(len(total_comp)):
            if total_comp[j].find(comps[i]) != -1:
                temp_c.append(total_comp[j])
                temp_v.append(total_vol[j])
                temp_cat.append(comps[i])
        cut_comp.append(temp_c)
        cut_vol.append(temp_v)
        cut_category.append(temp_cat)

    modified_data = pd.DataFrame(
    {'cut_comp': [value for l in cut_comp for value in l],
     'cut_vol': [value for l in cut_vol for value in l],
     'cut_category' : [value for l in cut_category for value in l]
    })

    sns.stripplot(x="cut_category", y="cut_vol", data=modified_data, jitter=True)
    sns.plt.show()
    return

# stripplot_comps(["Li", "Co", "Na"], 5000)

# shows energy_pa for two compositions
def show_energy(comp1, comp2, cut):
    count = cut
    isolation_var1 = comp1
    isolation_var2 = comp2

    comp = data.head(count).comp
    energy = data.head(count).energy_pa

    cut_comp1 = []
    cut_comp2 = []
    cut_energy1 = []
    cut_energy2 = []

    for i in range(len(comp)):
        # for first composition
        if comp[i].find(isolation_var1) != -1:
            cut_comp1.append(comp[i])
            cut_energy1.append(energy[i])
        if comp[i].find(isolation_var2) != -1:
            cut_comp2.append(comp[i])
            cut_energy2.append(energy[i])

    combined_comp = cut_comp1 + cut_comp2
    combined_energy = cut_energy1 + cut_energy2
    index = np.arange(len(combined_comp))
    plt.scatter(index, combined_energy, color="black", label=isolation_var1)
    plt.xticks(index, combined_comp, rotation=70)
    plt.legend()
    plt.show()

# show_energy("Li", "Mg", 300)
# import matminer
# from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval
# from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
#
# df_citrine = CitrineDataRetrieval().get_dataframe(formula='Si', property='band gap',
#                                                data_type='EXPERIMENTAL')
# df_mp = MPDataRetrieval().get_dataframe(criteria='Si', properties=['band_gap'])

# Electronegativity is a chemical property which describes how well an atom can attract an electron to itself. Values for electronegativity run from 0 to 4. Electronegativity is used to predict whether a bond between atoms will be ionic or covalent. It can also be used to predict if the resulting molecule will be polar or nonpolar. This table is a list of electronegativity values of the elements.

training_set = data
min_value = min(np.array(training_set["energy_pa"], dtype=float))
training_set.replace(str(min_value), np.nan, inplace=True)


training_set["is_groundstate"] = False
for gid,group in training_set.groupby('comp'):
    group = group.sort_values("energy_pa", ascending=True)
    training_set.loc[group.iloc[0].name, 'is_groundstate'] = True

original_count = len(training_set)
training_set = training_set[training_set['is_groundstate']]
removed = original_count - len(training_set)
print("Removed ", removed," / ",original_count)

from pymatgen import Composition
from matminer.descriptors.composition_features import ElementFractionAttribute

# def calc_attributes(training_set):
#     training_set_updated = ElementFractionAttribute().featurize_all(training_set, col_id="comp")

#     return training_set_updated

# all_desc = calc_attributes(training_set)
# print(np.shape(all_desc))










import pickle

#
# data = data.iloc[np.random.permutation(len(data))]

# pickle.dump(data, open("training_data.pkl", "wb"))


modified_data = pickle.load(open("training_data.pkl", "rb")).head(100)
modified_data.reset_index(inplace=True)
modified_data.drop('index', 1, inplace=True)


def get_vector_from_elements_list(L):
    """
    Given a list of element strings in format: [Li4, Sn2, Mg1, ...], this will return a vector where every index represents element occurance
    """
    vector = [0]*118
    for i in range(len(L)):
        element = L[i]
        # get symbol
        symbol = ''.join([i for i in element if not i.isdigit()])
        # get atomic number from symbol
        atomic_number_i = mg.Element(symbol).number-1

        n_moles = int(''.join([i for i in element if i.isdigit()]))
        vector[atomic_number_i] = n_moles
    return vector


def fill_training_data():
    lst_comp = []
    for i in range(len(modified_data.index)):
        lst_comp.append(mg.Composition(modified_data.iloc[i]['comp']))

    modified_data['weight'] = np.asarray([c.weight for c in lst_comp])
    modified_data['is_element'] = np.asarray([c.is_element for c in lst_comp])

    print(np.asarray([c.formula.split() for c in lst_comp]).size)

    modified_data['element_freq'] = np.asarray([c.formula.split() for c in lst_comp])

    modified_data['vector'] = modified_data['element_freq'].apply(get_vector_from_elements_list, 1)

    print(modified_data.head())


# fill_training_data()
