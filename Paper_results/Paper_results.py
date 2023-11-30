# -*- coding: utf-8 -*-
"""Pymoosh and Nevergrad

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Xk3gY1SK0xIivFRaCGMzsYrP_7RIbGmQ

## Pymoosh and Nevergrad for the inverse design of multilayered structures

This colab is based on Pymoosh (https://github.com/AnMoreau/PyMoosh) and Nevergrad (https://github.com/facebookresearch/nevergrad). If you like our codes, please consider putting a star on our githubs :-)

Context = shape optimization for photonics

# Here we do installations, imports and parametrizations. You might want to create your own copy of this colab (File, Save a copy). Edit here for parametrizing your run, before clicking on "run everything" in the "run" menu.
"""

#!pip install pymoosh
#!pip install bayes-optim==0.2.5.5
#!pip install nevergrad

# Let us get rid of some deprecation warning in SkLearn.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")  #, category=FutureWarning)

# Warning questions
def doint(s):
  return 7 + sum([ord(c)*i for i, c in enumerate(s)])

# Choose colors for algos
def get_color(o):
  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
  #c = {"CMA": "c", "QODE": "r", "BFGS": "g", "LBFGSB": "b", "GradBFGS": "k", "QNDE": "m"}[o]
  algs = ["CMA", "QODE", "BFGS", "LBFGSB", "GradBFGS", "QNDE"]
  if o in algs:
      return colors[algs.index(o)]
  return colors[doint(o) % len(colors)]

# Please modify the parameters below at your best convenience.
maximum_time_per_run_in_seconds = 60000  # Maximum number of seconds per run (then: interruption).
num_experiments = 31  # Number of times we reproduce each experiments.
maximum_optimization_overhead = 500  # Maximum ratio between the cost and the cost of the objective function only.

Init = -7.0
Grad = True
Full = False
Para = False
NoverD = 10
Factor = 0
# List of optimization methods in the extended setup.
list_optims = ["QODE", "GradBFGS", "GradLBFGSB", "BFGS", "LBFGSB", "ChainMetaModelSQP", "MetaModel", "SQP", "CMA", "DE", "RotatedTwoPointsDE", "OnePlusOne", "TwoPointsDE", "PSO", "GeneticDE", "NelderMead", "Cobyla" ,"Powell", "MemeticDE", "MetaModelDE"]  #,  "BO", "PCABO"]
list_optims = ["QODE", "GradBFGS", "GradLBFGSB", "BFGS", "LBFGSB", "CMA", "PSO", "DE", "OnePlusOne", "QNDE"]
list_optims = ["GradBFGS", "DE", "TwoPointsDE"]
list_optims = [ "TinyLhsDE", "TinyQODE", "ASCMADEthird", "MetaModelDE", "NeuralMetaModelDE", "SVMMetaModelDE", "RFMetaModelDE", "MetaModelTwoPointsDE", "NeuralMetaModelTwoPointsDE", "SVMMetaModelTwoPointsDE", "RFMetaModelTwoPointsDE", "GeneticDE", "MemeticDE", "QNDE", "GradBFGS", "TwoPointsDE", "DE"]
list_optims = ["QODE", "GradBFGS", "BFGS", "LBFGSB", "CMA", "PSO", "OnePlusOne", "QNDE", "SQOPSO", "ChainMetaModelSQP", "NelderMead", "Cobyla"]
list_optims = ["QODE", "QNDE", "GradBFGS", "BFGS", "CMA", "LBFGSB"] #"NgIoh4", "DiagonalCMA", "ChainMetaModelSQP", "Cobyla", "DE", "SQP"]
list_optims = ["QODE", "QNDE", "GradBFGS", "BFGS", "CMA"] #"NgIoh4", "DiagonalCMA", "ChainMetaModelSQP", "Cobyla", "DE", "SQP"]
if Full:
    list_optims = ["ChainDE", "QODE", "QNDE", "GradBFGS", "BFGS", "CMA", "LBFGSB", "NgIoh4", "DiagonalCMA", "ChainMetaModelSQP", "Cobyla", "DE", "SQP"]
if not Grad:
    list_optims = [l for l in list_optims if "Grad" not in l]
if Para:
    list_optims = [o for o in list_optims if o not in ["Cobyla", "SQP", "ChainMetaModelSQP", "GradBFGS"] and "BFGS" not in o]
# if you want more, you might add:  list_optims += ["BayesOptimBO", "PCABO", "RCobyla", "Shiwa", "CMandAS2", "NGOpt", "NGOptRW"]

# Choice of objective function. List of possibilities readable just below.
obj_name = "photovoltaics"
assert obj_name in ["bragg", "photovoltaics", "minibragg", "bigbragg", "bigphotovoltaics", "ellipsometry", "hugephotovoltaics"]
run_performance = True #True  # Whether we want to run the comparison between various optimization methods.

# Let us create a context: dimension, budget, objective_function, bounds.
#budget = int(Factor * {"bragg": 100000, "minibragg": 10000, "bigbragg": 100000, "photovoltaics": 10000, "bigphotovoltaics": 1500, "hugephotovoltaics": 1200, "ellipsometry": 100}[obj_name])
budget = int(Factor * {"bragg": 10000, "minibragg": 10000, "bigbragg": 100, "photovoltaics": 1000, "bigphotovoltaics": 600, "hugephotovoltaics": 1200, "ellipsometry": 100}[obj_name])
#budget = {"bragg": 10000, "bigbragg": 100, "photovoltaics": 1000, "bigphotovoltaics": 600, "ellipsometry": 100}[obj_name]

if obj_name == "bragg":
  nb_layers = 20
  opti_wave = 600
  mat1 = 1.4
  mat2 = 1.8
  min_th = 0 # We don't want negative thicknesses.
  max_th = opti_wave/(2*mat1) # A thickness of lambda/2n + t has the same behaviour as a thickness t

elif obj_name == "minibragg":
  nb_layers = 10
  opti_wave = 600
  mat1 = 1.4
  mat2 = 1.8
  min_th = 0 # We don't want negative thicknesses.
  max_th = opti_wave/(2*mat1) # A thickness of lambda/2n + t has the same behaviour as a thickness t

elif obj_name == "bigbragg":
  nb_layers = 40
  opti_wave = 600
  mat1 = 1.4
  mat2 = 1.8
  min_th = 0 # We don't want negative thicknesses.
  max_th = opti_wave/(2*mat1) # A thickness of lambda/2n + t has the same behaviour as a thickness t

elif obj_name == "ellipsometry":
  nb_layers = 1
  min_th = 50
  max_th = 150

elif obj_name == "photovoltaics":
  nb_layers = 10
  min_th = 30
  max_th = 250

elif obj_name == "bigphotovoltaics":
  nb_layers = 20
  min_th = 30
  max_th = 250

elif obj_name == "hugephotovoltaics":
  nb_layers = 32
  min_th = 30
  max_th = 250

else:
  assert False, f"Unknown objective function {obj_name}"

dim = {"bragg": nb_layers, "minibragg": nb_layers, "bigbragg": nb_layers, "bigphotovoltaics": nb_layers, "hugephotovoltaics": nb_layers, "photovoltaics": nb_layers, "ellipsometry": 2 * nb_layers, "ellipsometry2": 2 * nb_layers}[obj_name]
if Factor == 0.0:
   budget = NoverD *dim
if Factor < 0.0:
   budget = -Factor * 5000*dim

#if dim < 15:  # We remove Bayesian optimization from high-dimensional contexts.
#  list_optims += ["BO", "PCABO"]

# All problems.
min_ind = 1.1
max_ind = 3

context_string = f"We work on {obj_name}, dim={dim}, budget={budget}, bounds=[{min_th},{max_th}]"
print(context_string)

"""
# Here we define the Pymoosh objective functions."""

import PyMoosh as pm
import nevergrad as ng
import numpy as np
import matplotlib.pyplot as plt

def bragg(x):
  # This cost function corresponds to the problem
  # of maximizing the reflectance, at a given wavelength,
  # of a multilayered structure with alternating refractive
  # indexes. This problem is inspired by the first cases studied in
  # https://www.nature.com/articles/s41598-020-68719-3
  # :-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:
  # The input, x, corresponds to the thicknesses of all the layers, :
  # starting with the upper one.                                    :
  # :-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:
  x = list(x)
  n = len(x)
  # Working wavelength
  wl = 600.
  materials = [1,1.4**2,1.8**2]
  stack = [0] + [2,1] * (n//2) + [2]
  thicknesses = [0.] + list(x) + [0.]
  structure = pm.Structure(materials,stack,np.array(thicknesses),verbose = False)
  _, R = pm.coefficient_I(structure,wl,0.,0)
  cost = 1-R
  return cost

def photovoltaics(x):
  n = len(x)
  materials = [1., 2., 3., "SiA"]
  stack = [0] + [1,2] * (n//2) + [3]
  thicknesses = [0] + list(x) + [30000]
  structure = pm.Structure(materials, stack, np.array(thicknesses), verbose = False)
  incidence = 0
  pola = 0
  wl_min = 375
  wl_max = 750
  active_lay = len(thicknesses) - 1
  number_pts = 300
  eff, curr, curr_max, wl, spectrum, absorb4 = pm.photo(structure, incidence, pola, wl_min, wl_max, active_lay, number_pts)
  cost = 1 - eff
  return cost

mat = [1.] + [np.random.random()*(max_ind-min_ind) + min_ind for _ in range(nb_layers)] + ["Gold"]
layers = list(range(nb_layers+2))
structure = [0] + [np.random.random()*(max_th-min_th) + min_th for _ in range(nb_layers)] + [0]


angle = 40*np.pi/180
wav_list = np.linspace(400, 800, 100)

def ref_structure(mat, layers, structure, wav_list, angle):
    struct = pm.Structure(mat, layers, structure,verbose = False)
    ellips = np.zeros(len(wav_list), dtype=complex)
    for i, wav in enumerate(wav_list):
        r_s, _, _, _ = pm.coefficient(struct, wav, angle, 0)
        r_p, _, _, _ = pm.coefficient(struct, wav, angle, 1)

        ellips[i] = r_p/r_s
    return ellips, struct

ref_ellips, ellips_structure = ref_structure(mat, layers, structure, wav_list, angle)


def ellipsometry(X, ref_ellips=ref_ellips, wav_list=wav_list, angle=angle, nb_layers=nb_layers):
    mat = [1.] + [x for x in X[:nb_layers]] + ["Gold"]
    layers = [i for i in range(nb_layers+2)]
    structure = np.array([0] + [x for x in X[nb_layers:]] + [0])

    ellips = np.zeros(len(wav_list), dtype=complex)
    interface = pm.Structure(mat, layers, structure, verbose=False)
    for i, wav in enumerate(wav_list):
      try:
        r_s, _, _, _  = pm.coefficient_A(interface, wav, angle, 0)
        r_p, _, _, _  = pm.coefficient_A(interface, wav, angle, 1)
      except:
        r_s, _ = pm.coefficient_A(interface, wav, angle, 0)
        r_p, _ = pm.coefficient_A(interface, wav, angle, 1)

      ellips[i] = r_p/r_s
    val = np.sum(np.abs(ellips - ref_ellips))
    return val


objective_function = {"bragg": bragg, "minibragg": bragg, "bigbragg": bragg, "photovoltaics": photovoltaics, "bigphotovoltaics": photovoltaics, "hugephotovoltaics": photovoltaics, "ellipsometry": ellipsometry}[obj_name]

"""
# Simple running example for DE with Bragg"""

if "bragg" in obj_name and not run_performance:# and not run_performance:

  X_min = np.array([min_th]*nb_layers)
  X_max = np.array([max_th]*nb_layers)


  best, convergence = pm.differential_evolution(bragg, budget, X_min, X_max)

  plt.plot(convergence)
  plt.xlabel("Optimization step")
  plt.ylabel("Cost")
  plt.show()

  # Showing the final spectrum
  materials = [1,mat1**2,mat2**2]
  stack = [0] + [2,1] * (nb_layers//2) + [2]
  thicknesses = [0.] + [t for t in best] + [0.]

  bragg_mirror = [opti_wave / (4*np.sqrt(materials[2])), opti_wave / (4*np.sqrt(materials[1]))] * (nb_layers//2)
  bragg_th = [0.] + [t for t in bragg_mirror] + [0.]
  structure = pm.Structure(materials,stack,thicknesses,verbose = False)
  bragg_structure = pm.Structure(materials,stack,bragg_th,verbose = False)

  wavelengths = np.linspace(opti_wave-150, opti_wave+150, 300)
  spectrum = np.zeros_like(wavelengths)
  bragg_spectrum = np.zeros_like(wavelengths)
  for i, wav in enumerate(wavelengths):
    _,_,R,_ = pm.coefficient(structure,wav,0.,0)
    spectrum[i] = R
    _,_,R,_ = pm.coefficient(bragg_structure,wav,0.,0)
    bragg_spectrum[i] = R

 # plt.plot(wavelengths, spectrum, label="Optimized structure")
  best = min(convergence)
  plt.savefig(f"phy_dim{dim}_budget{budget}_{best}_bragg_cv.new.png", bbox_inches='tight')
  plt.savefig(f"phy_dim{dim}_budget{budget}_{best}_bragg_cv.new.svg", bbox_inches='tight')
  plt.clf()
  plt.plot(wavelengths, bragg_spectrum, label="Bragg mirror")
  plt.xlabel("Wavelength (nm)")
  plt.ylabel("Reflectivity")
  plt.legend()
  plt.show()

  plt.savefig(f"phy_dim{dim}_budget{budget}_{best}_bragg.new.png", bbox_inches='tight')
  plt.savefig(f"phy_dim{dim}_budget{budget}_{best}_bragg.new.svg", bbox_inches='tight')
  plt.clf()
  structure.plot_stack(wavelength=opti_wave)
  plt.savefig(f"phy_dim{dim}_budget{budget}_{best}_bragg_stack.new.png", bbox_inches='tight')
  plt.savefig(f"phy_dim{dim}_budget{budget}_{best}_bragg_stack.new.svg", bbox_inches='tight')
  plt.clf()
  bragg_structure.plot_stack(wavelength=opti_wave)
  plt.savefig(f"phy_dim{dim}_budget{budget}_{best}_bragg_structure.new.png", bbox_inches='tight')
  plt.savefig(f"phy_dim{dim}_budget{budget}_{best}_bragg_structure.new.svg", bbox_inches='tight')

"""# Simple running example for DE with Photovoltaics




"""


def photovoltaics_plot(best, str):
    # Showing the final spectrum
    materials = [1., 2., 3., "SiA"]
    stack = [0] + [1,2] * (len(best) // 2) + [3]
    thicknesses = [0.] + [t for t in best] + [30000.]

    structure = pm.Structure(materials, stack, thicknesses, verbose = False)
    incidence = 0
    pola = 0
    wl_min = 375
    wl_max = 750
    active_lay = len(thicknesses) - 1
    number_pts = 300
    eff, curr, curr_max, wl, spectrum, absorb_opt = pm.photo(structure, incidence, pola, wl_min, wl_max, active_lay, number_pts)

    plt.clf()
    plt.plot(wl, absorb_opt, label = "optimized absorption")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption")
    plt.legend()
    plt.show()
    plt.savefig(f"phy_new_pv_spectrum_{str}.new.png", bbox_inches='tight')
    plt.savefig(f"phy_new_pv_spectrum_{str}.new.svg", bbox_inches='tight')

    plt.clf()
    thicknesses_for_image = [30] + [t for t in best] + [30]
    structure_image = pm.Structure(materials, stack, thicknesses_for_image, verbose = False)
    structure_image.plot_stack()
    plt.savefig(f"phy_new_pv_structure_{str}.new.png", bbox_inches='tight')
    plt.savefig(f"phy_new_pv_structure_{str}.new.svg", bbox_inches='tight')
    plt.clf()

if "photovoltaics" in obj_name and not run_performance: # and not run_performance:

  X_min = np.array([min_th]*nb_layers)
  X_max = np.array([max_th]*nb_layers)

  best, convergence = pm.differential_evolution(photovoltaics, budget, X_min, X_max)
  photovoltaics_plot(best, f"pv_depymoosh_{budget}")


"""# Simple runnign example for DE with Ellipsometry"""

if obj_name in ["ellipsometry", "ellipsometry2"] and not run_performance:  # See parametrization at the top of the present colab.
  nb_runs = 20

  X_min = np.array([min_ind] * (nb_layers) +  [min_th] * (nb_layers))
  X_max = np.array([max_ind] * (nb_layers) +  [max_th] * (nb_layers))

  convs = []
  bests = []
  scores = []
  for i in range(nb_runs):
    best, convergence = pm.differential_evolution(ellipsometry, budget, X_min, X_max)
    bests.append(best)
    convs.append(convergence)
    scores.append(convergence[-1])

  order = np.argsort(scores)
  print(order)

  # Last structure
  mat = [1.] + [x for x in best[:nb_layers]] + ["Gold"]
  layers = [i for i in range(nb_layers+2)]
  structure = [0] + [x for x in best[nb_layers:]] + [0]
  best_struct = pm.Structure(mat, layers, structure)

  for i in range(nb_runs):
    plt.plot(convs[i])
  plt.xlabel("Optimization step")
  plt.ylabel("Cost")
  plt.show()

  # Best ever structure
  best = bests[order[0]]
  mat = [1.] + [x for x in best[:nb_layers]] + ["Gold"]
  layers = [i for i in range(nb_layers+2)]
  structure = [0] + [x for x in best[nb_layers:]] + [0]
  best_struct = pm.Structure(mat, layers, structure)


  ellips_structure.plot_stack()
  plt.savefig(f"phy_dim{dim}_budget{budget}_struc_ellips.new.png", bbox_inches='tight')
  plt.savefig(f"phy_dim{dim}_budget{budget}_struc_ellips.new.svg", bbox_inches='tight')
  best_struct.plot_stack()
  plt.savefig(f"phy_dim{dim}_budget{budget}_stack_ellips.new.png", bbox_inches='tight')
  plt.savefig(f"phy_dim{dim}_budget{budget}_stack_ellips.new.svg", bbox_inches='tight')

"""# Performance comparisons

Here we compare different optimization methods.
The list of these methods is in the top code block.
The objective function used in the comparison is specified in the top code block.
The next code block does the plottings.
"""

# In some cases we want to be fast and have a big difference between algorithms.
#budget = budget // 10  # NG experiments are run with lower budget sometimes

if run_performance:
  from joblib import Parallel, delayed;

  import time
  # Now we run all algorithms and we store results in dictionaries.
  scores = {}
  computational_cost = {}
  yval = {}
  for optim_name in list_optims:
   #for suffix in range(12):
    print("Running ", optim_name)
    scores[optim_name] = []
    computational_cost[optim_name] = []
    yval[optim_name] = []
    #for xp in range(num_experiments):
    def run_xp(xp):
      yvalo = []
      # We slightly randomize the upper bound, for checking the robustness.
      r = 0. if xp == 0 else (np.random.rand() - 0.5) * (max_th - min_th) * .1
      if obj_name == "ellipsometry":
        instrumentation = ng.p.Instrumentation(
            ng.p.Array(shape=(nb_layers,), lower=min_ind, upper=max_ind),
            ng.p.Array(shape=(nb_layers,), lower=min_th, upper=max_th - r),
        )
        array1 = ng.p.Array(shape=(nb_layers,), lower=min_ind, upper=max_ind)
        #array1.set_mutation((max_ind - min_ind) / 6.).set_standardized_data([-(max_ind - min_ind) / 2] * nb_layers)
        array2 = ng.p.Array(shape=(nb_layers,), lower=min_th, upper=max_th - r)
        #array2.set_mutation((max_th - min_th) / 6.).set_standardized_data([-(max_th - r - min_th) / 2] * nb_layers),
        instrumentation = ng.p.Instrumentation( array1, array2,)
      else:
        v = min_th + (max_th - min_th - r) * Init / 6.
        if Init < 0:
           init = [min_th + (max_th - min_th - r) * np.random.rand() for _ in range(dim)]
        else:
           init = [v] * dim
        instrumentation = ng.p.Array(init=np.array(init), lower=min_th, upper=max_th - r)
        #instrumentation = ng.p.Array(shape=(dim,), lower=min_th, upper=max_th - r)
        #if Init > 0.01:
          #instrumentation.set_mutation((max_th - min_th)/ (1 + 5. / Init)).set_standardized_data([-(max_th - r - min_th) / (1 + Init)]*dim)
        #  instrumentation.set_mutation((max_th - min_th)/ (1 + 5. / Init)).set_standardized_data([-(max_th - r - min_th) / (1 + Init)]*dim)
      #try:
      #    instrumentation.set_mutation((max_th - min_th)/ (1 + 5. / Init)).set_standardized_data([-(max_th - r - min_th) / (1 + Init)]*dim)
      #except:
      #    print("no rescale for ellipso!")
      if "Grad" in optim_name:
          budget_modifier = (dim + 1) // 2
          optim = ng.optimizers.registry[optim_name[4:]](instrumentation, budget * budget_modifier, num_workers=(30 if Para else 1))
      else:
          budget_modifier = 1
          optim = ng.optimizers.registry[optim_name](instrumentation, budget * budget_modifier, num_workers=(30 if Para else 1))
      if Init < -7 and not "BFGS" in optim_name:
          for _ in range(30):
              optim.suggest([min_th + (-min_th + max_th - r) * np.random.rand() for _ in range(dim)])
      best_y = float("inf")
      t0 = time.time()
      xval = []
      obj_time = float("inf")
      for k in range(int(budget * budget_modifier)):
        if time.time() - t0 < min(maximum_time_per_run_in_seconds, maximum_optimization_overhead * (k+1) * obj_time):
          x = optim.ask()
          t1 = time.time()
          val = x.value
          if obj_name == "ellipsometry":
            val = list(val[0][0]) + list(val[0][1])
          elif obj_name == "ellipsometry2":
            val = [val[0][i][0] for i in range(len(val[0]))]
          y = objective_function(val)
          obj_time = float(time.time() - t1)
          optim.tell(x, y)
          if y < best_y:
            best_y = y
            best_x  = x
        if (k + 1) % budget_modifier != 0:
            continue
        kk = k // budget_modifier
        if int(np.log2(kk + 1) + .999999) == int(np.log2((kk + 1))) or kk == budget - 1:
          xval += [kk+1]
          if len(yvalo) < len(xval):
            #yval[optim_name] += [[]]
            yvalo += [[]]
          #yval[optim_name][len(xval)-1] += [best_y]
          yvalo[len(xval)-1] = [best_y]
      #print(best_y)
      if "photovol" in obj_name:
          print("drawing a pv structure")
          photovoltaics_plot(best_x.value, f"{obj_name}_{best_y}_ng_{budget}_{optim_name}_{np.random.randint(500)}")
      return (xval, best_y, time.time() - t0, yvalo)
      #computational_cost[optim_name] += [time.time() - t0]
      #scores[optim_name] += [best_y]
    results = Parallel(n_jobs=num_experiments)(delayed(run_xp)(xp) for xp in range(num_experiments))
    yval[optim_name] = []
    for a, r in enumerate(results):
      xval = r[0]
      assert len(xval) == len(r[3]), f"{len(xval)} != {len(r[3][0])}" 
      for i, x in enumerate(r[3]):
          assert len(x) == 1
          if len(yval[optim_name]) < i+1:
              yval[optim_name] += [[]]
          yval[optim_name][i] += [[float(x[0])]]
      computational_cost[optim_name] += [float(r[2])]
      scores[optim_name] += [float(r[1])]

"""# Below a code block for plotting the results."""

if run_performance:
  import matplotlib.pyplot as plt
  plt.rcParams['font.size'] = "20" #str(int(plt.rcParams['font.size']) + 2)
  plt.rcParams['figure.figsize'] = [15, 9]
  plt.rcParams["figure.autolayout"] = True
  #from google.colab import files

  angles = [c * 2 * 3.14159 / 10 for c in range(10)] + [0.]

  # Plotting the convergence curves, NOT aggregated.
  for subcase in [11]: #range(32):
      plt.clf()
      np.random.seed(7)
      for o in sorted(scores, key=lambda x: -np.sum(scores[x]) / len(scores[x])):
        if np.random.RandomState(doint(o) + subcase).choice([True, False, False]) or (("QODE" == o or "GradBFGS" == o) and np.random.RandomState(doint(o) + subcase).choice([True, True, True, False])):
          pass  # Shame on me
        else:
          continue
        print(o, xval, yval[o])
        yval_mean = [np.sum(yval[o][i]) / len(yval[o][i]) for i in range(len(yval[o]))]
        print(yval_mean)
        yval_std = [np.std(yval[o][i]) / np.sqrt(len(yval[o][i]) - 1) for i in range(len(yval[o]))]
        try:
          #c = {"CMA": "c", "QODE": "r", "BFGS": "g", "LBFGSB": "b", "GradBFGS": "k", "QNDE": "m"}[o]
          c = get_color(o)
          print(f"{o} got {c}")
        except:
          c = np.random.RandomState(doint(o)).choice(["r", "g", "b", "k", "m", "c"])
        s = np.random.RandomState(doint(o)).choice([6, 10, 13])
        m = np.random.RandomState(doint(o)).choice(["o", "^", "v", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"])
        try:
            plt.semilogx(xval, yval_mean, label=o, c=c, marker=m, markersize=s)
        except Exception as e:
            assert False, f"{e}: xval={xval}, yval_mean={yval_mean}, label={o}"
        for k in range(len(yval[o][0])):
          yval_k = [yval[o][i][k] for i in range(len(yval[o]))]
          plt.semilogx(xval, yval_k, c=c, marker=m, markersize=s)
        #plt.annotate(o, (xval[-1], yval_mean[-1]), rotation=30)
        for i in range(len(xval)):
          #plt.plot([xval[i], xval[i]], [yval_mean[i] - yval_std[i], yval_mean[i] + yval_std[i]], c="k", marker="+")
          plt.semilogx([xval[i] * (1. + .1 * np.cos(theta)) for theta in angles], [yval_mean[i] + yval_std[i] * np.sin(theta) for theta in angles], c="grey", linestyle="dashed")
      plt.legend()
      plt.title(context_string + "\n Convergence curves")
      plt.tight_layout(pad=12.0, w_pad=12.0, h_pad=12.0)
      plt.savefig("fcc" + obj_name + f"_{Para}_{NoverD}_{str(list_optims).replace(',','_').replace(' ' ,'_')}_Factor" + str(Factor) + "_Init" + str(Init) + "_" + str(subcase) + "_" + ("grad.smalltinynew.png" if Grad else ".smalltinynew.png"), bbox_inches='tight')
      plt.savefig("fcc" + obj_name + f"_{Para}_{NoverD}_{str(list_optims).replace(',','_').replace(' ' ,'_')}_Factor" + str(Factor) + "_Init" + str(Init) + "_" + str(subcase) + "_" + ("grad.smalltinynew.svg" if Grad else ".smalltinynew.svg"), bbox_inches='tight')
      #files.download("fcc" + obj_name + ("grad.png" if Grad else ".png"))
      plt.pause(0.05)

      # Plotting the convergence curves, aggregated.
      plt.clf()
      np.random.seed(7)
      for o in sorted(scores, key=lambda x: -np.sum(scores[x]) / len(scores[x])):
        #print(o, xval, yval[o])
        yval_mean = [np.sum(yval[o][i]) / len(yval[o][i]) for i in range(len(yval[o]))]
        yval_std = [np.std(yval[o][i]) / np.sqrt(len(yval[o][i]) - 1) for i in range(len(yval[o]))]
        try:
          #c = {"CMA": "c", "QODE": "r", "BFGS": "g", "LBFGSB": "b", "GradBFGS": "k", "QNDE": "m"}[o]
          c = get_color(o)
          print(f"{o} got {c}")
        except:
          c = np.random.RandomState(doint(o)).choice(["r", "g", "b", "k", "m", "c"])
        s = np.random.RandomState(doint(o)).choice([6, 10, 13])
        m = np.random.RandomState(doint(o)).choice(["o", "^", "v", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"])
        plt.semilogx(xval, yval_mean, label=o, c=c, marker=m, markersize=s)
        #plt.annotate(o, (xval[-1], yval_mean[-1]), rotation=30)
        for i in range(len(xval)):
          #plt.plot([xval[i], xval[i]], [yval_mean[i] - yval_std[i], yval_mean[i] + yval_std[i]], c="k", marker="+")
          plt.semilogx([xval[i] * (1. + .1 * np.cos(theta)) for theta in angles], [yval_mean[i] + yval_std[i] * np.sin(theta) for theta in angles], c="grey", linestyle="dashed")
      plt.legend()
      plt.title(context_string + "\n Convergence curves")
      plt.tight_layout(pad=12.0, w_pad=12.0, h_pad=12.0)
      plt.savefig("cc" + obj_name + f"_{Para}_{NoverD}_{str(list_optims).replace(',','_').replace(' ' ,'_')}_Factor" + str(Factor) + "_Init" + str(Init) + "_" + str(subcase) + "_" + ("grad.smalltinynew.png" if Grad else ".smalltinynew.png"), bbox_inches='tight')
      plt.savefig("cc" + obj_name + f"_{Para}_{NoverD}_{str(list_optims).replace(',','_').replace(' ' ,'_')}_Factor" + str(Factor) + "_Init" + str(Init) + "_" + str(subcase) + "_" + ("grad.smalltinynew.svg" if Grad else ".smalltinynew.svg"), bbox_inches='tight')
      #files.download("cc" + obj_name + ("grad.png" if Grad else ".png"))
      plt.pause(0.05)

  # Now pretty plotting with score on the x-axis and comp. time on the y-axis.
  plt.clf()
  maxi = max([np.sum(scores[o]) / len(scores[o]) for o in scores])
  mini = min([np.sum(scores[o]) / len(scores[o]) for o in scores])
  width = maxi - mini
  for i, o in enumerate(sorted(scores, key=lambda x: np.sum(computational_cost[x]) / len(scores[x]))):
    avg_score = np.sum(scores[o]) / len(scores[o])
    std_score = np.std(scores[o]) / np.sqrt(len(scores[o]))
    avg_time = np.sum(computational_cost[o]) / len(computational_cost[o])
    print(o, avg_score, "+-", std_score, "   in time ", avg_time)
    plt.annotate("     " + o, (avg_score, avg_time), rotation=0. + 0. * i / len(scores))
    plt.semilogx([avg_score], [avg_time], '*')
    plt.semilogx([avg_score-std_score, avg_score+std_score], [avg_time, avg_time], '-')
    plt.xlabel("Score")
    plt.ylabel("Comp. time")
  plt.title(context_string + "\n Loss vs computational cost")
  plt.tight_layout(pad=12.0, w_pad=12.0, h_pad=12.0)
  plt.savefig("comptime" + obj_name + f"_{Para}_{NoverD}_{str(list_optims).replace(',','_').replace(' ' ,'_')}_Factor" + str(Factor) + "_Init" + str(Init) + "_" + ("grad.smalltinynew.png" if Grad else ".smalltinynew.png"), bbox_inches='tight')
  plt.savefig("comptime" + obj_name + f"_{Para}_{NoverD}_{str(list_optims).replace(',','_').replace(' ' ,'_')}_Factor" + str(Factor) + "_Init" + str(Init) + "_" + ("grad.smalltinynew.svg" if Grad else ".smalltinynew.svg"), bbox_inches='tight')
  #files.download("comptime" + obj_name + ("grad.png" if Grad else ".png"))
  plt.pause(0.05)

  # Boxplots for all algorithms.
  plt.clf()
  names = sorted(scores.keys(), key=lambda x: np.sum(scores[x]) / len(scores[x]))
  plt.boxplot([scores[o] for o in names])
  x = []
  y = []
  for u, k in enumerate(names):
    print(f"Algorithm {u+1}/{len(names)}: {k}")
    for i, s in enumerate(scores[k]):
      x += [u + 0.9 + np.random.rand() * .2]
      y += [sorted(scores[k])[i]]
  plt.plot(x, y, "*")
  import pylab
  ax = plt.gca()
  ax.set_xticklabels(ax.get_xticks(), rotation = 45, ha="right")
  pylab.xticks(range(1, len(names)+1), names)
  plt.title(f"Performance of different algorithms for {obj_name}\n with budget {budget} and dim {dim}\n (lower the better)")
  plt.tight_layout(pad=12.0, w_pad=12.0, h_pad=12.0)
  plt.grid(True)

  plt.savefig("bp" + obj_name + f"_{Para}_{NoverD}_{str(list_optims).replace(',','_').replace(' ' ,'_')}_Factor" + str(Factor) + "_Init" + str(Init) + "_" + ("grad.smalltinynew.png" if Grad else ".smalltinynew.png"), bbox_inches='tight')
  plt.savefig("bp" + obj_name + f"_{Para}_{NoverD}_{str(list_optims).replace(',','_').replace(' ' ,'_')}_Factor" + str(Factor) + "_Init" + str(Init) + "_" + ("grad.smalltinynew.svg" if Grad else ".smalltinynew.svg"), bbox_inches='tight')
  #files.download("bp" + obj_name + ("grad.png" if Grad else ".png"))
  plt.pause(0.05)

  # And consistency plots for all algorithms.
  plt.clf()
  names = sorted(scores.keys(), key=lambda x: -np.max(scores[x]))
  for u, k in enumerate(names):
    c = get_color(k)
    s = np.random.RandomState(doint(k)).choice([6, 10, 13])
    m = np.random.RandomState(doint(k)).choice(["o", "^", "v", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"])
    print(f"Risky Algorithm {len(names)-u}/{len(names)}: {k}")
    plt.plot(range(len(scores[k])), sorted(scores[k], key=lambda x: x), label=k, c=c, marker=m, markersize=s) #, marker=np.random.choice(["o", "^", "v", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"]))
  plt.legend()
  plt.title(f"{context_string}\nScores of the different runs of each method, sorted. The lower the better.")
  plt.tight_layout(pad=12.0, w_pad=12.0, h_pad=12.0)
  plt.grid(True)
  plt.savefig("q" + obj_name + f"_{Para}_{NoverD}_{str(list_optims).replace(',','_').replace(' ' ,'_')}_Factor" + str(Factor) + "_Init" + str(Init) + "_" + ("grad.smalltinynew.png" if Grad else ".smalltinynew.png"), bbox_inches='tight')
  plt.savefig("q" + obj_name + f"_{Para}_{NoverD}_{str(list_optims).replace(',','_').replace(' ' ,'_')}_Factor" + str(Factor) + "_Init" + str(Init) + "_" + ("grad.smalltinynew.svg" if Grad else ".smalltinynew.svg"), bbox_inches='tight')
  #files.download("q" + obj_name + ("grad.png" if Grad else ".png"))
