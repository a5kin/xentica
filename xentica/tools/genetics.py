"""A collection of functions allowing genetics manipulations."""

from xentica import core
from xentica.tools import xmath


def genome_crossover(state, num_genes, *genomes, rng_name="rng"):
    """
    Crossover given genomes in stochastic way.

    :param state:
        A container holding model's properties.
    :param num_genes:
        Genome length, assuming all genomes has same number of genes.
    :param genomes:
        A list of genomes (integers) to crossover
    :param rng_name:
        Name of ``RandomProperty``.

    :returns: Single integer, a resulting genome.

    """
    gene_choose = core.IntegerVariable()
    new_genome = core.IntegerVariable()
    num_genomes = core.IntegerVariable()
    for gene in range(num_genes):
        gene_choose *= 0
        num_genomes *= 0
        for genome in genomes:
            gene_choose += ((genome >> gene) & 1 & (genome > 0)) << num_genomes
            num_genomes += (genome > 0)
        rand_val = getattr(state, rng_name).uniform
        winner_gene = xmath.int(rand_val * num_genomes)
        new_genome += ((gene_choose >> winner_gene) & 1) << gene
    return new_genome
