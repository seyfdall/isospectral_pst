# Isospectral Unfolding and Substitution Experiments

This repository collects exploratory work on the inverse problem for isospectral reductions of graphs: given a reduced graph or scalar reduction function, can one reconstruct or characterize larger graphs that preserve the same reduction?

The work blends symbolic graph algebra, brute-force search, weighted-path constructions, Mathematica-based conjecture testing, and ML-style reconstruction experiments. This formed the research and experimental basis for the thesis: A Graphical Approach to Isospectral Unfoldings.

## Core idea

For a graph G with a retained vertex set S, the isospectral reduction is the Schur complement

$$
R(\lambda) = M_{SS} - M_{S\bar S}(M_{\bar S\bar S} - \lambda I)^{-1}M_{\bar S S}.
$$

The main theme of the project is to study the inverse direction: find admissible unfoldings G' for which the reduction matches a given target function. In many cases, the same reduction admits multiple non-isomorphic unfoldings, so the project also investigates enumeration, equivalence, and classification.

## Repository layout

### unfolding/

This is the main research area.

#### weighted_path_expansion.ipynb

This notebook focuses on weighted path constructions and equitable-partition style unfoldings. The computational pattern is:

- parameterize path-like or layered graph expansions,
- enumerate valid weighted combinations,
- check whether the resulting structure satisfies the reduction constraints,
- collect admissible unfoldings produced by path-based families.

In practical terms, this is a combinatorial search notebook for structured unfoldings that are easier to reason about than arbitrary graphs.

#### unfolding_mathematica/

This folder contains Mathematica notebooks used for symbolic exploration and conjecture checking. The notebooks implement:

- adjacency-matrix utilities,
- reduction/smash functions,
- equitable-partition and pattern search experiments,
- automorphism and same-reduction checks,
- testing of structural conjectures about how reductions behave under unfolding.

The notebooks are conceptual tools for probing whether candidate patterns always preserve or recover the right spectral behavior.

#### unfolding_ml/

This directory moves from exact symbolic work toward reconstruction and learning. It includes notebooks and benchmark scripts for:

- algebraic reconstruction of admissible unfoldings,
- exact symbolic verification of candidate graphs,
- numerical and polynomial-root approaches,
- gradient-based and retrieval-style model experiments,
- benchmarking of CPU symbolic versus GPU numeric search strategies.

The emphasis here is on whether a model can recover a graph from a reduction signature, and how to evaluate the result against exact isospectral admissibility.

### isospectral_substitution/

This folder contains lower-level combinatorial and arithmetic experiments on substitution/decomposition patterns. The code explores how parameter tuples satisfy system-level constraints that arise in reduction-based constructions. Broadly, it is a search-and-decomposition toolkit for understanding which integer or symbolic blocks can be assembled into valid isospectral structures.

This material complements the unfolding notebooks by providing algebraic and decomposition heuristics for building or validating candidate configurations.

## Typical workflow

The project is organized around a few recurring stages:

1. Compute a target reduction from a known graph.
2. Search over candidate complement graphs or structured expansions.
3. Assemble full graphs from admissible components.
4. Verify isospectral equivalence by exact symbolic or numerical reduction checks.
5. Deduplicate by graph isomorphism and compare families of unfoldings.
6. Use ML or fast search methods when exact enumeration becomes too large.

## Research focus

The repository is best understood as a toolkit for studying the non-uniqueness of inverse isospectral unfoldings. A key message is that reductions do not determine a unique full graph; instead, they often admit multiple distinct unfoldings. The project explores how to find, classify, and recover these families.

## Practical note

This is an exploratory research repository rather than a polished software package. The notebooks and scripts are designed for experimentation, symbolic verification, and pattern discovery rather than production deployment.
