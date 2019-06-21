"""The collection of functions for genetics manipulations."""

from xentica import core
from xentica.tools import xmath


def genome_crossover(state, num_genes, *genomes, max_genes=None,
                     mutation_prob=0, rng_name="rng"):
    """
    Crossover given genomes in stochastic way.

    :param state:
        A container holding model's properties.
    :param num_genes:
        Genome length, assuming all genomes has the same number of genes.
    :param genomes:
        A list of genomes (integers) to crossover
    :param max_genes:
        Upper limit for '1' genes in the resulting genome.
    :param mutation_prob:
        Probability of a single gene's mutation.
    :param rng_name:
        Name of ``RandomProperty``.

    :returns: Single integer, a resulting genome.

    """
    max_genes = max_genes or num_genes
    gene_choose = core.IntegerVariable()
    new_genome = core.IntegerVariable()
    num_genomes = core.IntegerVariable()
    num_active = core.IntegerVariable()
    new_gene = core.IntegerVariable()
    rand_val = getattr(state, rng_name).uniform
    start_gene = core.IntegerVariable()
    start_gene *= 0
    start_gene += xmath.int(rand_val * num_genes) % len(genomes)
    for gene in range(num_genes):
        gene_choose *= 0
        num_genomes *= 0
        gene = (gene + start_gene) % num_genes
        for genome in genomes:
            gene_choose += ((genome >> gene) & 1 & (genome > 0)) << num_genomes
            num_genomes += (genome > 0)
        rand_val = getattr(state, rng_name).uniform
        winner_gene = xmath.int(rand_val * num_genomes)
        new_gene *= 0
        new_gene += ((gene_choose >> winner_gene) & 1)
        num_active += new_gene
        new_gene *= num_active <= max_genes
        is_mutated = 0
        if mutation_prob > 0:
            is_mutated = getattr(state, rng_name).uniform < mutation_prob
            is_mutated = is_mutated * (gene_choose > 0)
        new_genome += (new_gene ^ is_mutated) << gene
    return new_genome
