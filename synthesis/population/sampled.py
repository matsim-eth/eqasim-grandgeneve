import pandas as pd




def configure(context):
    context.stage("synthesis.population.sampled_before_spatial_selection")
    context.stage("synthesis.population.spatial.home.zones")
    context.stage("data.spatial.iris")

    context.config("hts", default = "emp")
    if context.config("hts") == "edgt_74":
        context.config("edgt74_version", default = "adisp")


def execute(context):
    population = context.stage("synthesis.population.sampled_before_spatial_selection")
    homes      = context.stage("synthesis.population.spatial.home.zones")

    if context.config("hts") == "edgt_74":
        if context.config("edgt74_version") == "adisp":
            iris = context.stage("data.spatial.iris")

            homes = homes.merge(iris[["iris_id", "edgt_area"]], on = "iris_id", how = "left")
            population = population.merge(homes[["household_id", "edgt_area"]], on = "household_id", how = "left")

            print(len(population))
            population = population[population["edgt_area"].notna()]
            print(len(population))

            print(population.groupby("edgt_area", dropna = False)["person_id"].count())

    return population

