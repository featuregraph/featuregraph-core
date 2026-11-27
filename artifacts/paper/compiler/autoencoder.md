Introduction

Autoencoders and other commonly-used feature extraction methods often do not make assumptions about the physical shapes of the signals they extract features from. As a result, the features, while representative of the signal, can have meanings that are not interpretable by humans.

In this paper, we investigate whether representations that incorporate prior knowledge of physical signal behavior, originally designed for human and computational agent consumption, can also support multiple downstream learning tasks without modifying the underlying behavioral construction. 

To investigate this question, we use FeatureGraph, a representation framework that assumes prior knowledge of the type of physical behavior to be represented, but not the specific shape of the signal. Once the behavioral construction is specified, features are extracted automatically from the observed signal without requiring external context or domain-specific information.

<contributions of this study, close out introduction>

Section 1: Representation framework development

The representation framework was originally designed to summarize signal properties for human and computational agent consumption [cite Cohere post]. Its purpose was to shift the work of signal analysis from a language model to a deterministic program while allowing the human user to interact naturally with the language model to retrieve those results. The agent layer currently allows agents to translate research questions expressed in natural language into structured queries, which are validated against the rules of a predefined query language and executed against a table of signal measurements. 

The framework's development was not motivated by a need to construct features for machine learning. However, subsequent experimentation revealed that the representations produced by FeatureGraph could also serve as effective inputs to machine learning models. This unexpected finding motivated the present study.

Section 1a: Behavioral constructions

FeatureGraph provides reusable constructions for two classes of physical behavior: oscillation, characterized by recurring waveforms, and accumulation, characterized by the addition of physical quantities over time. These constructions operate solely on the observed structure of the signal, without requiring knowledge of its physical meaning. The same construction logic can therefore be applied to signals across different scientific and engineering domains.

Once a construction is chosen and its parameters specified, individual features are extracted automatically. Each feature has an explicit analytical meaning established by the construction and does not need to be inferred after extraction. The current study is restricted to oscillation behaviors.

Section 1b: Intended use

The original interest of this study was to reduce the amount of analysis work a language model would have to do by shifting that work to deterministic programs, while still allowing the entire process to be automated. 






