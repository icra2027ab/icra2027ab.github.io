"""
core
----
High-level pipeline that orchestrates the four main phases:
  1. retrieval   – find relevant in-distribution demos from the dataset
  2. abstraction – extract feature trends from the retrieved demos
  3. alignment   – monotonically align live execution to the abstracted demo
  4. guidance    – translate aligned feature trends into policy steering gradients
"""
