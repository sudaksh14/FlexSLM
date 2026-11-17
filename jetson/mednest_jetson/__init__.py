"""MedNest on the Jetson Orin Nano: elastic (one-weights, many-widths) inference and its benchmarks.

elastic    ElasticModel: every trained width as a zero-copy view of the full-width weights
measure    system-memory poller, page-cache drop, width loading from the single full checkpoint
workload   stratified test subset, prompts/scoring as on the cluster, timed greedy decoding
"""
