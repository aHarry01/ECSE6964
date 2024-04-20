import numpy as np
from pulp import LpMaximize, LpMinimize, LpProblem, lpSum, LpVariable

def closest_subset_target_sum(numbers, target_sum):
    # modified from: https://stackoverflow.com/questions/71034015/subset-sum-problem-for-a-possible-closest-value-to-the-target-sum-in-python-for
    indices = range(len(numbers))
    variables = LpVariable.dicts("Choice", indices, cat="Binary")

    leq_prob = LpProblem('leq', LpMaximize)
    geq_prob = LpProblem('geq', LpMinimize)

    leq_prob += lpSum([variables[i]*numbers[i] for i in indices]) <= target_sum
    leq_prob += lpSum([variables[i]*numbers[i] for i in indices])

    geq_prob += lpSum([variables[i]*numbers[i] for i in indices]) >= target_sum
    geq_prob += lpSum([variables[i]*numbers[i] for i in indices])

    leq_prob.solve()
    leq_indices = []
    leq_choices = []
    for i in indices:
        if variables[i].value() == 1:
            leq_indices.append(i)
            leq_choices.append(numbers[i])

    if sum(leq_choices) == target_sum:
        solution = leq_choices
        solution_indices = leq_indices
    else:
        geq_prob.solve()
        geq_indices = []
        geq_choices = []
        for i in indices:
            if variables[i].value() == 1:
                geq_indices.append(i)
                geq_choices.append(numbers[i])

        if target_sum-sum(leq_choices) <= sum(geq_choices)-target_sum:
            solution = leq_choices
            solution_indices = leq_indices
        else:
            solution = geq_choices
            solution_indices = geq_indices

    return solution_indices, solution
