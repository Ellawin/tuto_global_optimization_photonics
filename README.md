# Global optimization of photonic structures with `PyMoosh` and `Nevergrad`

Tutorial python notebooks, accompanying the paper *An illustrated tutorial on global optimization in nanophotonics* by P. Bennet et al. ([arXiv:2309.09760](https://arxiv.org/abs/2309.09760)).

We demonstrate how to do global optimization of the geometry of multi-layered photonic structures, 2D gratings and plasmonic nanostructures. Multi-layer simulations are done with [`PyMoosh`](https://github.com/AnMoreau/PyMoosh), 2D gratings with a home-made RCWA code and plasmonic nanostructures with [`pyGDM`](https://homepages.laas.fr/pwiecha/pygdm_doc/). As optimization toolkit we use [`Nevergrad`](https://facebookresearch.github.io/nevergrad/). For more information about PyMoosh, our Python-based simulation library designed to provide a comprehensive set of numerical tools allowing to compute essentially all optical characteristics of multilayered structures, please consult our recent ['PyMoosh tutorial'](https://arxiv.org/pdf/2309.00654.pdf).

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

You can run the Notebook directly on your own computer, or you can connect to the Colab plateform to use external server if you want more computational resources.

  - `01_simple_Bragg_optimization_with_pymoosh.ipynb`: Very simple tutorial how to use `PyMoosh`'s internal DE optimizer ([link to google colab version](https://drive.google.com/file/d/1d9J9dqxvhBc88QPPP3WDT-4-YOOPIi5a/view?usp=sharing))
  
  - `02_simple_optimization_with_nevergrad.ipynb`: Very simple tutorial how to use `Nevergrad` optimizers for photonics ([link to google colab version](https://drive.google.com/file/d/1fRvlXTPhfa7cmDQ-xozGLIEBBZIxhP2w/view?usp=sharing))
  
  - `03_benchmark_optimization_with_nevergrad.ipynb`: Tutorial how to benchmark several algorithms, demonstrated on a small Bragg mirror problem ([link to google colab version](https://drive.google.com/file/d/1q3y9DA87kR7-Vip0ie3A5ioiwmrAr3cY/view?usp=sharing))

  - '04_simple_Ellipso_optimization.ipynb' : Tutorial setting up the ellipsometry problem ([link to google colab version](https://drive.google.com/file/d/1JbnoOo6xvA_5LDBwCuF00xXaTjlQ-r8V/view?usp=sharing))

  - '05_simple_Photovoltaics_optimization.ipynb' : Tutorial setting up the photovoltaics problem ([link to google colab version](https://drive.google.com/file/d/1MHzrVjxxHiy8s813vokzQABpnAo1Q9pr/view?usp=sharing))
  
  - `06_simple_2Dgrating_optimization_with_Nevergrad.ipynb`: Tutorial setting up the 2D grating problem ([link to google colab version](https://drive.google.com/file/d/16-61bLnL-dVe6jfG3Hqcv_pOk8QP1M2l/view?usp=sharing))

In this example we reproduce the optimization of a 2D grating to minimize its specular blue reflection (450 nm). Each layer of the structure is composed of one block of dielectric in air. Each block is characterized by its width, thickness and position in a period of 600 nm. The width, thickness and position of each blocks are subject to the optimizer. The optical properties are computed with a RCWA method. For more information about this optimization, please consult the work described in [Barry et al. Sci Rep 10, 12024 (2020)](https://www.nature.com/articles/s41598-020-68719-3#citeas).
  
  - `07_simple_3D_plasmonic_nanostructure_directional.ipynb`: Tutorial setting up the 3D nanoantenna problem ([link to google colab version](https://drive.google.com/file/d/1HVzmNqhjNy-XLUP_Qwq8g4jkrhxGpjU2/view?usp=sharing))

In this example we reproduce the optimization of a directional plasmonic nanoantenna that couples with a local quantum emitter, and steers the light towards a defined solid angle as described in [Wiecha et al. Opt. Express 27, pp. 29069, (2019)](https://doi.org/10.1364/OE.27.029069).

The optimization target is to maximize the ratio of emission towards a specific target solid angle and emission towards the rest of the sphere surface. The emitting system is a local oscillating dipole at fixed wavelength (800 nm), representing for example a quantum dot, coupled to a gold nanostructure, for simplicity of the tutorial, in vacuum.
The optimization acts on the geometry of the gold nanostructure, which is constituted of 40 small gold cubes, each of 40x40x40 nm^3. Their X, Y positions are discretized in steps of these blocks, and subject to the optimizer.


## Benchmark results

The figures from our paper *An illustrated tutorial on global optimization in nanophotonics* by P. Bennet et al. ([arXiv:2309.09760](https://arxiv.org/abs/2309.09760)) are generated with the 'Paper_results.py' program. 


