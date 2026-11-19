## Framework

Nazia Habib

November 2026

## Introduction

One line of work in temporal abstraction, beginning with Shahar's 1997 framework, can be mapped through three stages:

1. Declared knowledge. States are qualitatively defined by classification and persistence rules given by experts for a specific domain (Shahar 1994 and 1997).
2. Cutoffs learned from labels. Cutpoints are chosen to best separate outcome classes (Moskovitch and Shahar 2015). The method covers state abstraction only in its experiments, not gradients or interpolation. Knowledge-based cutoffs as a comparison baseline.
3. Declarations derived from the signal and recorded. Windows and cutoffs can come from measured properties of the signal such as the period [cite]. Construction choices such as accumulation threshold that are not derived can be supplied as explicit variables. The declaration is recorded with its provenance and sensitivity can be tested for each chosen parameter.

Acquiring expert declarations for state classification and persistence is costly [cite]. For densely sampled signals, much of this information can be derived from the signal itself and recorded. Densely sampled signals are now routinely collected and openly available, and the derivations straightforward to code using standard data analysis libraries. 

This paper maps FeatureGraph, a framework for constructing objects from time-series signals, onto the KBTA framework (Shahar 1997), identifying overlap between the two methods and what each lacks relative to the other. It organizes object construction into five layers (primitive, declaration, object, measure, scientific interpretation), and shows evidence that declarations derived from a signal can be recorded and tested.
