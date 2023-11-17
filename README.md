# Global optimization of photonic structures with `PyMoosh` and `Nevergrad`

Tutorial python notebooks, accompanying the paper *An illustrated tutorial on global optimization in nanophotonics* by P. Bennet et al. ([arXiv:2309.09760](https://arxiv.org/abs/2309.09760)).

We demonstrate how to do global optimization of the geometry of multi-layered photonic structures, 2D gratings and plasmonic nanostructures. Multi-layer simulations are done with [`PyMoosh`](https://github.com/AnMoreau/PyMoosh), 2D gratings with a home-made RCWA code and plasmonic nanostructures with [`pyGDM`](https://homepages.laas.fr/pwiecha/pygdm_doc/). As optimization toolkit we use [`Nevergrad`](https://facebookresearch.github.io/nevergrad/).

## Goal of the example problems

**The tutorials have two main goals:**

  1. Providing a simple introduction and a starting point for global optimization of multi-layer structures.
  
  2. Describing how to apply global optimization to different problems in nano-photonics
  

**Five specific applications are treated:**

  1. Optimization of a Bragg mirror.
  
  2. Solving of an ellipsometry inverse problem.
  
  3. Design of a sophisticated antireflection coating to optimize solar absorption in a photovoltaic solar cell.
  
  4. Optimization of a 2D grating for maximum specular reflectance in the first diffraction order at a given wavelength.
  
  5. Design of a plasmonic nanostructure for directional emission from a local emitter.


## List of all notebooks

  - `01_simple_optimization_with_pymoosh.ipynb`: Very simple tutorial how to use `PyMoosh`'s internal DE optimizer ([link to google colab version](https://drive.google.com/file/d/1ECBCpJWD3uIRMPHOHduIyokkSxB_osnX/view?usp=sharing))
  
  - `02_simple_optimization_with_nevergrad.ipynb`: Very simple tutorial how to use `Nevergrad` optimizers for photonics ([link to google colab version](https://drive.google.com/file/d/1aUQDEfG43Bxm08mm5IHgYVT985LWBNa0/view?usp=sharing))
  
  - `03_benchmark_optimization_with_nevergrad.ipynb`: Tutorial how to benchmark several algorithms, demonstrated on a small Bragg mirror problem ([link to google colab version](https://drive.google.com/file/d/1L30YtJgbq5dVf-ZpIYPmuwSNMKmUp_uW/view?usp=sharing))
  
  - `04_simple_2Dgrating_optimization_with_Nevergrad.ipynb`: Tutorial setting up the ellipsometry problem ([link to google colab version](https://drive.google.com/file/d/1Nz_UDWTI0xIuYwx2KkTJHQI51xbdGUJy/view?usp=sharing))
  
  - `05_simple_3D_plasmonic_nanostructure_directional.ipynb`: Algorithm benchmark on the ellipsometry problem ([link to google colab version](https://colab.research.google.com/drive/1kDp0o1Mqaq-bdf8RoxCH6ALZhGukyHNz?usp=sharing))

## Benchmark results

The figures from our paper *An illustrated tutorial on global optimization in nanophotonics* by P. Bennet et al. ([arXiv:2309.09760](https://arxiv.org/abs/2309.09760)) are generated with the 'Paper_results.py' program. 

